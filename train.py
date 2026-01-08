import os
import torch
import wandb
from tqdm import tqdm

# import constant_learnign rate swith warm up
import logging
import math

from transformers import (
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from tqdm.auto import tqdm

from model import Pix2SeqModel
from dataloader import get_pix2seq_dataloaders
from tokenizer import Pix2SeqTokenizer
import torchvision
from utils import fast_map50

from torchmetrics.detection.mean_ap import MeanAveragePrecision


@torch.no_grad()
def sample_images(
    model, val_loader, tokenizer, accelerator, device, global_step, num_samples=8
):
    """
    Samples a batch of images, generates predictions, draws GT (Green) and Pred (Red) boxes,
    and logs them to WandB.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    logging.info(f"[Step {global_step}] Sampling images for visualization...")

    class_names = tokenizer.class_names

    # 1. Get a single batch of data (Handle Dictionary)
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        val_loader_iter = iter(val_loader)
        batch = next(val_loader_iter)

    images = batch["images"].to(device)[:num_samples]
    target_tokens = batch["tokens"].to(device)[:num_samples]
    valid_sizes = batch["valid_sizes"][:num_samples]

    current_batch_size = images.size(0)

    if current_batch_size == 0:
        return

    # 2. Generate Predictions (Greedy Search or Nucleus)
    generated_seqs = unwrapped_model.generate(images, max_new_tokens=100)

    visualizations = []

    # --- PREPARE INVERSE NORMALIZATION CONSTANTS ---
    # We move these to CPU because we pull images.cpu() inside the loop
    inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 3. Iterate through samples to draw boxes
    for i in range(current_batch_size):
        img_tensor = images[i].cpu()

        # --- DENORMALIZE START ---
        # Formula: original = (normalized * std) + mean
        img_tensor = img_tensor * inv_std + inv_mean

        # Clamp to [0, 1] range to avoid uint8 overflow/underflow artifacts
        img_tensor = torch.clamp(img_tensor, 0, 1)
        # --- DENORMALIZE END ---

        img_to_draw = (img_tensor * 255).to(torch.uint8)

        # Get the VALID dimensions for correct scaling
        h_valid, w_valid = valid_sizes[i].tolist()

        # --- A. Process Ground Truth (GREEN) ---
        gt_boxes_norm, gt_label_indices = tokenizer.decode(target_tokens[i])
        gt_boxes_abs = []
        gt_labels_str = []
        for box, lbl_idx in zip(gt_boxes_norm, gt_label_indices):
            # Scale by VALID dimensions
            xb = box[0] * w_valid
            yb = box[1] * h_valid
            wb = box[2] * w_valid
            hb = box[3] * h_valid

            if wb > 0 and hb > 0:
                gt_boxes_abs.append([xb, yb, xb + wb, yb + hb])
                gt_labels_str.append(f"GT: {class_names[lbl_idx]}")

        if gt_boxes_abs:
            img_to_draw = torchvision.utils.draw_bounding_boxes(
                image=img_to_draw,
                boxes=torch.tensor(gt_boxes_abs, dtype=torch.float),
                labels=gt_labels_str,
                colors="green",
                width=3,
                font_size=12,
            )

        # --- B. Process Predictions (RED) ---
        pred_boxes_norm, pred_label_indices = tokenizer.decode(generated_seqs[i])
        pred_boxes_abs = []
        pred_labels_str = []
        for box, lbl_idx in zip(pred_boxes_norm, pred_label_indices):
            # Scale by VALID dimensions
            xb = box[0] * w_valid
            yb = box[1] * h_valid
            wb = box[2] * w_valid
            hb = box[3] * h_valid

            if wb > 1.0 and hb > 1.0:
                pred_boxes_abs.append([xb, yb, xb + wb, yb + hb])
                safe_idx = max(0, min(lbl_idx, len(class_names) - 1))
                pred_labels_str.append(f"Pred: {class_names[safe_idx]}")

        if pred_boxes_abs:
            try:
                img_to_draw = torchvision.utils.draw_bounding_boxes(
                    image=img_to_draw,
                    boxes=torch.tensor(pred_boxes_abs, dtype=torch.float),
                    labels=pred_labels_str,
                    colors="red",
                    width=2,
                    font_size=12,
                )
            except ValueError:
                pass

        visualizations.append(
            wandb.Image(img_to_draw, caption=f"Step {global_step} - Sample {i}")
        )

    if accelerator.is_main_process and visualizations:
        accelerator.log({"val/visualizations": visualizations}, step=global_step)

    model.train()



@torch.no_grad()
def validate(model, val_loader, tokenizer, accelerator, device, global_step):
    """
    A lightweight 'Proxy' validation to check convergence speed.
    NOT comparable to official COCO mAP.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    # Increased slightly to avoid cutting off crowded scenes
    VAL_MAX_TOKENS = 150  
    
    # Trackers for Distributed Averaging
    local_score_sum = 0.0
    local_count = 0

    if accelerator.is_main_process:
        print(f"\n[Step {global_step}] Validation...")

    # Iterate
    for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
        images = batch["images"].to(device)
        target_tokens = batch["tokens"].to(device)
        valid_sizes = batch["valid_sizes"] # [B, 2]

        # Generate
        generated = unwrapped_model.generate(images, max_new_tokens=VAL_MAX_TOKENS)

        # Process Batch
        for i in range(images.size(0)):
            h_valid, w_valid = valid_sizes[i].tolist()

            # Decode
            pred_boxes_n, pred_labels = tokenizer.decode(generated[i])
            gt_boxes_n, gt_labels = tokenizer.decode(target_tokens[i])

            # Skip images with no GT (backgrounds) to avoid div/0 or skewing
            if len(gt_boxes_n) == 0:
                continue

            # Convert to Absolute Coords (Validation needs pixels, not 0-1)
            # Handle empty predictions gracefully
            if len(pred_boxes_n) > 0:
                pred_boxes = torch.tensor(pred_boxes_n, device=device)
                # Scale: x*w, y*h, w*w, h*h (assuming box format is xywh or similar)
                # NOTE: Ensure your tokenizer returns [x,y,w,h]. 
                # If using xyxy, logic differs. Assuming standard [x,y,w,h] normalized:
                pred_boxes[:, 0] *= w_valid
                pred_boxes[:, 1] *= h_valid
                pred_boxes[:, 2] *= w_valid
                pred_boxes[:, 3] *= h_valid
                
                # Convert [x,y,w,h] -> [x1,y1,x2,y2] for IoU calculation
                pred_boxes[:, 2] += pred_boxes[:, 0]
                pred_boxes[:, 3] += pred_boxes[:, 1]
            else:
                pred_boxes = torch.empty((0, 4), device=device)

            # Do same for GT
            gt_boxes = torch.tensor(gt_boxes_n, device=device)
            gt_boxes[:, 0] *= w_valid
            gt_boxes[:, 1] *= h_valid
            gt_boxes[:, 2] *= w_valid
            gt_boxes[:, 3] *= h_valid
            gt_boxes[:, 2] += gt_boxes[:, 0]
            gt_boxes[:, 3] += gt_boxes[:, 1]

            pred_labels = torch.tensor(pred_labels, device=device)
            gt_labels = torch.tensor(gt_labels, device=device)

            # --- CALCULATE METRIC ---
            score = fast_map50(pred_boxes, pred_labels, gt_boxes, gt_labels)
            
            local_score_sum += score
            local_count += 1

    # We turn our scalars into tensors so Accelerator can gather them
    stats_tensor = torch.tensor([local_score_sum, local_count], device=device)
    
    # Sum up the scores and counts from ALL GPUs
    stats_tensor = accelerator.reduce(stats_tensor, reduction="sum")
    
    total_score = stats_tensor[0].item()
    total_count = stats_tensor[1].item()

    # Avoid div/0
    avg_map = total_score / max(total_count, 1)

    if accelerator.is_main_process:
        print(f"Fast Proxy mAP@50: {avg_map:.4f}")
        accelerator.log({"val/proxy_map_50": avg_map}, step=global_step)

    model.train()
    return avg_map


def resume_from_checkpoint(device, filename, model, optim, scheduler):
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    global_step = checkpoint["step"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logging.info(f"Resumed from checkpoint: {filename} at step {global_step}")

    return global_step


def save_ckpt(args, accelerator, model, optim, scheduler, global_step, filename):
    if not args.save_intermediate_models:
        filename = os.path.join(args.ckpt_saved_dir, "final-model.pth")

    checkpoint = {
        "step": global_step,
        "model_state_dict": accelerator.get_state_dict(model),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
    }
    accelerator.save(checkpoint, filename)
    logging.info("Saving checkpoint: %s ...", filename)


def train(args):
    global_step = 0

    # setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
    )

    accelerator.init_trackers(
        project_name=args.project_name,
        # add kwargs for wandb
        init_kwargs={"wandb": {"config": vars(args)}},
    )

    # set device
    device = accelerator.device
    # model
    model = Pix2SeqModel(
        dino_model_name=args.dino_model_name,
        decoder_dim=args.decoder_dim,
        num_decoder_layers=args.num_decoder_layers,
        nhead=args.nhead,
        dim_head=args.dim_head,
        max_seq_len=args.max_seq_len,
    )

    tokenizer = Pix2SeqTokenizer(num_bins=args.num_bins)

    # Train loders
    train_dl, val_dl = get_pix2seq_dataloaders(
        root_path=args.root, batch_size=args.batch_size, num_workers=args.num_workers
    )


    print("DataLoaders prepared.")
    print(len(train_dl))

    # training parameters
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05
    )

    steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    num_training_steps = args.num_epochs * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    # scheduler = None

    # prepare model, optimizer, and dataloader for distributed training
    model, optim, scheduler, train_dl, val_dl = accelerator.prepare(
        model, optim, scheduler, train_dl, val_dl
    )

    print(len(train_dl))

    # load models
    if args.resume:
        global_step = resume_from_checkpoint(
            device, args.resume, model, optim, scheduler
        )

    effective_steps_per_epoch = math.ceil(
        len(train_dl) / args.gradient_accumulation_steps
    )
    effective_training_steps = args.num_epochs * effective_steps_per_epoch

    logging.info(
        f"Effective batch size per device: {args.batch_size * args.gradient_accumulation_steps}"
    )
    logging.info(f"Effective steps per epoch: {effective_steps_per_epoch}")
    logging.info(f"Effective Total training steps: {effective_training_steps}")

    start_epoch = global_step // effective_training_steps

    model.train()
    for epoch in range(start_epoch, args.num_epochs):
        with tqdm(
            train_dl, dynamic_ncols=True, disable=not accelerator.is_main_process
        ) as train_dl:
            for batch in train_dl:
                images, targets = batch["images"], batch["tokens"]
                loss_masks = batch["loss_masks"]

                images = images.to(device)
                targets = targets.to(device)

                with accelerator.accumulate(model):
                    optim.zero_grad(set_to_none=True)

                    with accelerator.autocast():
                        loss = model(
                            images, target_tokens=targets, loss_masks=loss_masks
                        )[0]

                    accelerator.backward(loss)

                    if accelerator.sync_gradients and args.max_grad_norm:
                        accelerator.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    optim.step()
                    if scheduler is not None:
                        scheduler.step()

                # =========================
                # LOGGING AND CHECKPOINTING
                # =========================
                if accelerator.sync_gradients:
                    # Checkpointing: Only Main Process
                    if (not (global_step % args.ckpt_every)):
                        if accelerator.is_main_process:
                            save_ckpt(
                                args,
                                accelerator,
                                model,
                                optim,
                                scheduler,
                                global_step,
                                os.path.join(
                                    args.ckpt_saved_dir,
                                    f"{wandb.run.name}-step-{global_step}.pth",
                                ),
                            )

                        accelerator.wait_for_everyone()

                    # Sampling: Main Process Works, Everyone Waits
                    if not (global_step % args.sample_every):

                        # Work
                        if accelerator.is_main_process:
                            sample_images(
                                model,
                                val_dl,
                                tokenizer,
                                accelerator,
                                device,
                                global_step,
                                num_samples=10,
                            )

                        accelerator.wait_for_everyone()

                    #  Validation: Run on ALL processes
                    if not (global_step % args.eval_every) and global_step != 0 and args.resume is None:
                        validate(
                            model, val_dl, tokenizer, accelerator, device, global_step
                        )
                        accelerator.wait_for_everyone()

                    #  Logging Scalars: Main Process Only
                    if accelerator.is_main_process:
                        log_dict = {
                            "loss": loss.item(),
                            "lr": optim.param_groups[0]["lr"],
                        }
                        accelerator.log(log_dict, step=global_step)

                    global_step += 1
                    args.resume = None  # only skip eval for first iteration if resuming

    # save the final model
    if accelerator.is_main_process:
        save_ckpt(
            accelerator,
            model,
            optim,
            scheduler,
            global_step,
            os.path.join(args.ckpt_saved_dir, f"{wandb.run.name}-final.pth"),
        )

    accelerator.end_training()
    print("Train finished!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # project / dataset
    parser.add_argument(
        "--project_name", type=str, default="Pix2Seq", help="WandB project name"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/datasets/coco-dataset/coco",
        help="Path to dataset",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size per device"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--num_bins", type=int, default=1000, help="Number of bins for box encoding"
    )

    # model parameters
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="facebook/dinov2-base",
        help="DINO model name",
    )
    parser.add_argument(
        "--decoder_dim", type=int, default=512, help="Decoder model dimension"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=6, help="Number of decoder layers"
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads in the decoder"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length for the decoder",
    )
    parser.add_argument(
        "--dim_head",
        type=int,
        default=64,
        help="Dimension of each attention head in the decoder",
    )

    # training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=400, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="LR warmup steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )

    # logging / checkpointing
    parser.add_argument(
        "--ckpt_every", type=int, default=20000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--save_intermediate_models",
        default=False,
        action="store_true",
        help="Whether to save intermediate models during training",
    )
    parser.add_argument(
        "--eval_every", type=int, default=20000, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=2000,
        help="Sample and log reconstructions every N steps",
    )
    parser.add_argument(
        "--ckpt_saved_dir", type=str, default="ckpt", help="Directory to save outputs"
    )

    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    kwargs = vars(args)
    print("Training configuration:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    train(args)
