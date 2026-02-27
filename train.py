import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import wandb
from tqdm import tqdm
# import constant_learnign rate swith warm up
from torch.optim.lr_scheduler import CyclicLR
import logging
import math

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import os
import torch
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from einops import rearrange
import logging

from model import Pix2SeqModel
from dataloader import get_pix2seq_dataloaders
from tokenizer import Pix2SeqTokenizer
import torchvision

from torchmetrics.detection.mean_ap import MeanAveragePrecision



@torch.no_grad()
def sample_images(model, val_loader, tokenizer, accelerator, device, global_step, num_samples=8):
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
                gt_boxes_abs.append([xb, yb, xb+wb, yb+hb])
                gt_labels_str.append(f"GT: {class_names[lbl_idx]}")

        if gt_boxes_abs:
            img_to_draw = torchvision.utils.draw_bounding_boxes(
                image=img_to_draw,
                boxes=torch.tensor(gt_boxes_abs, dtype=torch.float),
                labels=gt_labels_str,
                colors="green", width=3, font_size=12
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
                pred_boxes_abs.append([xb, yb, xb+wb, yb+hb])
                safe_idx = max(0, min(lbl_idx, len(class_names) - 1))
                pred_labels_str.append(f"Pred: {class_names[safe_idx]}")

        if pred_boxes_abs:
            try:
                img_to_draw = torchvision.utils.draw_bounding_boxes(
                    image=img_to_draw,
                    boxes=torch.tensor(pred_boxes_abs, dtype=torch.float),
                    labels=pred_labels_str,
                    colors="red", width=2, font_size=12
                )
            except ValueError:
                pass 

        visualizations.append(wandb.Image(img_to_draw, caption=f"Step {global_step} - Sample {i}"))

    if accelerator.is_main_process and visualizations:
        accelerator.log({"val/visualizations": visualizations}, step=global_step)

    model.train()

@torch.no_grad()
def validate(model, val_loader, tokenizer, accelerator, device, global_step):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    
    # Increase max_new_tokens for better recall during validation
    VAL_MAX_TOKENS = 300 
    
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False).to(device)
    
    if accelerator.is_main_process:
        print(f"\n[Step {global_step}] Starting Validation...")
    
    # Iterate over the loader (handling dictionary)
    for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
        
        images = batch["images"].to(device)
        target_tokens = batch["tokens"].to(device)
        valid_sizes = batch["valid_sizes"] # Keep CPU or move to device if needed
        
        # 1. Generate
        generated_seqs = unwrapped_model.generate(images, max_new_tokens=VAL_MAX_TOKENS)
        
        preds = []
        targets = []
        
        for i in range(images.size(0)):
            # Get valid dimensions for this image
            h_valid, w_valid = valid_sizes[i].tolist()
            
            # --- Process Prediction ---
            pred_boxes_norm, pred_labels = tokenizer.decode(generated_seqs[i])
            
            pred_boxes_abs = []
            for box in pred_boxes_norm:
                # Scale by VALID dimensions
                pb = [box[0] * w_valid, box[1] * h_valid, box[2] * w_valid, box[3] * h_valid]
                # xywh -> xyxy
                pb_xyxy = [pb[0], pb[1], pb[0] + pb[2], pb[1] + pb[3]]
                pred_boxes_abs.append(pb_xyxy)
            
            if len(pred_boxes_abs) > 0:
                preds.append({
                    'boxes': torch.tensor(pred_boxes_abs, dtype=torch.float32, device=device),
                    'scores': torch.ones(len(pred_labels), device=device),
                    'labels': torch.tensor(pred_labels, dtype=torch.long, device=device)
                })
            else:
                preds.append({
                    'boxes': torch.tensor([], device=device),
                    'scores': torch.tensor([], device=device),
                    'labels': torch.tensor([], device=device)
                })

            # --- Process Ground Truth ---
            gt_boxes_norm, gt_labels = tokenizer.decode(target_tokens[i])
            
            gt_boxes_abs = []
            for box in gt_boxes_norm:
                # Scale by VALID dimensions
                gb = [box[0] * w_valid, box[1] * h_valid, box[2] * w_valid, box[3] * h_valid]
                gb_xyxy = [gb[0], gb[1], gb[0] + gb[2], gb[1] + gb[3]]
                gt_boxes_abs.append(gb_xyxy)
            
            targets.append({
                'boxes': torch.tensor(gt_boxes_abs, dtype=torch.float32, device=device),
                'labels': torch.tensor(gt_labels, dtype=torch.long, device=device)
            })
            
        metric.update(preds, targets)
        
    result = metric.compute()
    
    map_50 = result['map_50'].item()
    map_75 = result['map_75'].item()
    map_coco = result['map'].item()
    
    if accelerator.is_main_process:
        print(f"Validation Results - mAP (COCO): {map_coco:.4f} | mAP@50: {map_50:.4f} | mAP@75: {map_75:.4f}")
        
        accelerator.log({
            "val/mAP": map_coco,
            "val/mAP_50": map_50,
            "val/mAP_75": map_75
        }, step=global_step)
        
    model.train()
    return map_coco



def resume_from_checkpoint(device, filename, model, optim, scheduler):
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        global_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
        
        logging.info(f"Resumed from checkpoint: {filename} at step {global_step}")

        return global_step


def save_ckpt(args, accelerator, model, optim, scheduler, global_step, filename):
    if not args.save_intermediate_models:
        filename = os.path.join(args.ckpt_saved_dir, f'final-model.pth')
    else:
        filename = os.path.join(args.ckpt_saved_dir, f'{wandb.run.name}-step-{global_step}.pth')

    checkpoint={
            'step': global_step,
            'model_state_dict': accelerator.get_state_dict(model),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
        }
    accelerator.save(checkpoint, filename)
    logging.info("Saving checkpoint: %s ...", filename)




def train(args):

    global_step = 0

    # setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb")

    accelerator.init_trackers(
            project_name=args.project_name,
            # add kwargs for wandb
            init_kwargs={"wandb": {
                "config": vars(args)
            }}	
    )

    # set device
    device = accelerator.device
    # model
    model = Pix2SeqModel()

    tokenizer = Pix2SeqTokenizer(num_bins=1000)
    
    # Train loders
    train_dl, val_dl = get_pix2seq_dataloaders(
        root_path=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


    # training parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05
        )

    steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    num_training_steps = args.num_epochs * steps_per_epoch
   
    scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )


    # scheduler = None

    # prepare model, optimizer, and dataloader for distributed training
    model, optim, scheduler, train_dl, val_dl = accelerator.prepare(
        model, 
        optim, 
        scheduler, 
        train_dl, 
        val_dl
    )
    
    # load models
    if args.resume:
        global_step = resume_from_checkpoint(device, args.resume, model, optim, scheduler)

    effective_steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    effective_training_steps = args.num_epochs * effective_steps_per_epoch

    logging.info(f"Effective batch size per device: {args.batch_size * args.gradient_accumulation_steps}")
    logging.info(f"Effective steps per epoch: {effective_steps_per_epoch}")
    logging.info(f"Effective Total training steps: {effective_training_steps}")

    start_epoch = global_step // effective_training_steps

    model.train()
    for epoch in range(start_epoch, args.num_epochs):
        with tqdm(train_dl, dynamic_ncols=True, disable=not accelerator.is_main_process) as train_dl:
            for (batch) in train_dl:
                
                images , targets = batch["images"], batch["tokens"]
                loss_masks = batch["loss_masks"] 

                images = images.to(device)
                targets = targets.to(device)

                with accelerator.accumulate(model):
                    
                    optim.zero_grad(set_to_none=True)
                    
                    with accelerator.autocast():
                        loss = model(images, target_tokens=targets, loss_masks=loss_masks)[0]
                        
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients and args.max_grad_norm:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    optim.step()	
                    if scheduler is not None:
                        scheduler.step()

            
                # =========================
                # LOGGING AND CHECKPOINTING
                # =========================
                if accelerator.sync_gradients:
                    
                    # Checkpointing: Only Main Process
                    if not (global_step % args.ckpt_every):
                        if accelerator.is_main_process:
                            save_ckpt(args, accelerator, model, optim, scheduler, global_step,
                                    os.path.join(args.ckpt_saved_dir, f'{wandb.run.name}-step-{global_step}.pth'))
                            
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
                                num_samples=10
                            )
                      
                        accelerator.wait_for_everyone()

                    #  Validation: Run on ALL processes
                    if not (global_step % args.eval_every) and global_step != 0:
                        validate(
                            model,
                            val_dl,
                            tokenizer,
                            accelerator,
                            device,
                            global_step
                        )
                        accelerator.wait_for_everyone() 
                    
                    #  Logging Scalars: Main Process Only
                    if accelerator.is_main_process:
                        log_dict = {
                            "loss": loss.item(),
                            "lr": optim.param_groups[0]['lr']
                        }
                        accelerator.log(log_dict, step=global_step)
                    
                    global_step += 1


    # save the final model
    if accelerator.is_main_process:
        save_ckpt(
            accelerator,
            model,
            optim,
            scheduler,
            global_step,
            os.path.join(args.ckpt_saved_dir, f'{wandb.run.name}-final.pth')
        )

    accelerator.end_training()        
    print("Train finished!")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()

    # project / dataset
    parser.add_argument('--project_name', type=str, default='Pix2Seq', help="WandB project name")
    parser.add_argument('--root', type=str, default='/home/pranoy/datasets/coco',help="Path to dataset")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loader workers")

    # training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=2000, help="LR warmup steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], help="Mixed precision training mode")


    # logging / checkpointing
    parser.add_argument('--ckpt_every', type=int, default=20000, help="Save checkpoint every N steps")
    parser.add_argument('--save_intermediate_models', default=False, action='store_true', help="Whether to save intermediate models during training")
    parser.add_argument('--eval_every', type=int, default=20000, help="Evaluate every N steps")
    parser.add_argument('--sample_every', type=int, default=2000, help="Sample and log reconstructions every N steps")
    parser.add_argument('--ckpt_saved_dir', type=str, default='ckpt', help="Directory to save outputs")

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
