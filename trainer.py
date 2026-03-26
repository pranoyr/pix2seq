from __future__ import annotations


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

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    logging.info(f"[Step {global_step}] Sampling images for visualization...")

    class_names = tokenizer.class_names

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

    generated_seqs = unwrapped_model.generate(images, max_new_tokens=100)

    visualizations = []


    inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


    for i in range(current_batch_size):
        img_tensor = images[i].cpu()

        img_tensor = img_tensor * inv_std + inv_mean
        
      
        img_tensor = torch.clamp(img_tensor, 0, 1)

        img_to_draw = (img_tensor * 255).to(torch.uint8)
        

        h_valid, w_valid = valid_sizes[i].tolist()

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

        pred_boxes_norm, pred_label_indices = tokenizer.decode(generated_seqs[i])
        pred_boxes_abs = []
        pred_labels_str = []
        for box, lbl_idx in zip(pred_boxes_norm, pred_label_indices):
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
    
    for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
        
        images = batch["images"].to(device)
        target_tokens = batch["tokens"].to(device)
        valid_sizes = batch["valid_sizes"] 
        
        generated_seqs = unwrapped_model.generate(images, max_new_tokens=VAL_MAX_TOKENS)
        
        preds = []
        targets = []
        
        for i in range(images.size(0)):
            h_valid, w_valid = valid_sizes[i].tolist()
            
            pred_boxes_norm, pred_labels = tokenizer.decode(generated_seqs[i])
            
            pred_boxes_abs = []
            for box in pred_boxes_norm:
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

          
            gt_boxes_norm, gt_labels = tokenizer.decode(target_tokens[i])
            
            gt_boxes_abs = []
            for box in gt_boxes_norm:
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


def save_ckpt(accelerator, model, optim, scheduler, global_step, filename, save_intermediate_models, ckpt_saved_dir):
    if not save_intermediate_models:
        filename = os.path.join(ckpt_saved_dir, f'final-model.pth')
    else:
        filename = os.path.join(ckpt_saved_dir, f'{wandb.run.name}-step-{global_step}.pth')

    checkpoint={
            'step': global_step,
            'model_state_dict': accelerator.get_state_dict(model),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
        }
    accelerator.save(checkpoint, filename)
    logging.info("Saving checkpoint: %s ...", filename)




class Pix2SeqTrainer:
    def __init__(self, 
                model : nn.Module,
                tokenizer : Pix2SeqTokenizer,
                train_dl : DataLoader,
                val_dl: DataLoader,
                optim : torch.optim.Optimizer,
                scheduler : torch.optim.lr_scheduler._LRScheduler,

                num_epochs: int = 100,
                batch_size: int = 16,
                gradient_accumulation_steps: int = 1,
                max_grad_norm: float = 1.0,

                ckpt_every: int = 100,
                eval_every: int = 1000,
                sample_every: int = 500,    
                save_intermediate_models: bool = False,
                ckpt_saved_dir: str = 'ckpt',
                resume: str | None = None,
                accelerator_kwargs: dict | None = None

                ):

        self.global_step = 0

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with="wandb"
        )

        self.accelerator.init_trackers(
            **accelerator_kwargs,
        )

        device = self.accelerator.device

        self.model, self.optim, self.scheduler, self.train_dl, self.val_dl = self.accelerator.prepare(
            model, 
            optim, 
            scheduler, 
            train_dl, 
            val_dl
        )

        self.tokenizer = tokenizer
        
        # load models
        if resume:
            self.global_step = resume_from_checkpoint(device, resume, self.model, self.optim, self.scheduler)

        effective_steps_per_epoch = math.ceil(len(self.train_dl) / gradient_accumulation_steps)
        effective_training_steps = num_epochs * effective_steps_per_epoch

        logging.info(f"Effective batch size per device: {batch_size * gradient_accumulation_steps}")
        logging.info(f"Effective steps per epoch: {effective_steps_per_epoch}")
        logging.info(f"Effective Total training steps: {effective_training_steps}")

        self.start_epoch = self.global_step // effective_steps_per_epoch
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.ckpt_every = ckpt_every
        self.eval_every = eval_every
        self.sample_every = sample_every
        self.save_intermediate_models = save_intermediate_models
        self.ckpt_saved_dir = ckpt_saved_dir



    @property
    def device(self):
        return self.accelerator.device

    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.num_epochs):
            with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
                for (batch) in train_dl:
                    
                    images , targets = batch["images"], batch["tokens"]
                    loss_masks = batch["loss_masks"] 

                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    with self.accelerator.accumulate(self.model):
                        
                        self.optim.zero_grad(set_to_none=True)
                        
                        with self.accelerator.autocast():
                            loss = self.model(images, target_tokens=targets, loss_masks=loss_masks)[0]
                            
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients and self.max_grad_norm:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        self.optim.step()	
                        if self.scheduler is not None:
                            self.scheduler.step()

        
                    if self.accelerator.sync_gradients:
                        # checkpointing
                        if not (self.global_step % self.ckpt_every):
                            if self.accelerator.is_main_process:
                                save_ckpt(self.accelerator, 
                                            model, 
                                            self.optim,
                                            self.scheduler,
                                            self.global_step,
                                            os.path.join(self.ckpt_saved_dir, f'{wandb.run.name}-step-{self.global_step}.pth'),
                                            self.save_intermediate_models,
                                            self.ckpt_saved_dir
                                            )

                            self.accelerator.wait_for_everyone()
                        
                        # sampling
                        if not (self.global_step % self.sample_every):
                            if self.accelerator.is_main_process:
                                sample_images(
                                    model,
                                    val_dl,
                                    tokenizer,
                                    self.accelerator,
                                    self.device,
                                    self.global_step,
                                    num_samples=10
                                )
                        
                            self.accelerator.wait_for_everyone()

                        # validate
                        if not (self.global_step % self.eval_every) and self.global_step != 0:
                            validate(
                                model,
                                val_dl,
                                tokenizer,
                                self.accelerator,
                                self.device,
                                self.global_step
                            )
                            self.accelerator.wait_for_everyone() 
                        
                        # logging
                        if self.accelerator.is_main_process:
                            log_dict = {
                                "loss": loss.item(),
                                "lr": self.optim.param_groups[0]['lr']
                            }
                            self.accelerator.log(log_dict, step=self.global_step)
                        
                        self.global_step += 1


        # save the final model
        if self.accelerator.is_main_process:
            save_ckpt(
                self.accelerator,
                model,
                self.optim,
                self.scheduler,
                self.global_step,
                os.path.join(self.ckpt_saved_dir, f'{wandb.run.name}-final.pth')
            )

        self.accelerator.end_training()        
        print("Train finished!")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()

    # project / dataset
    parser.add_argument('--project_name', type=str, default='Pix2Seq', help="WandB project name")
    parser.add_argument('--root', type=str, default='/mnt/datasets/coco-dataset/coco',help="Path to dataset")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loader workers")

    # training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=2000, help="LR warmup steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], help="Mixed precision training mode")
    parser.add_argument('--beta1', type=float, default=0.9, help="Adam beta1")
    parser.add_argument('--beta2', type=float, default=0.95, help="Adam beta2")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Adam weight decay")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")


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

    
    model = Pix2SeqModel()
    tokenizer = Pix2SeqTokenizer(num_bins=1000)
    
    # loaders
    train_dl, val_dl = get_pix2seq_dataloaders(
        root_path=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # optimizer 
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
        )

    steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    num_training_steps = args.num_epochs * steps_per_epoch
    
    # scheduler 
    scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    training_params = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm
    }

    logging_params = {
        "ckpt_every": args.ckpt_every,
        "eval_every": args.eval_every,
        "sample_every": args.sample_every,
        "save_intermediate_models": args.save_intermediate_models,
        "ckpt_saved_dir": args.ckpt_saved_dir,
        "resume": args.resume,
    }

    accelerator_kwargs={
            "project_name": args.project_name,
            "init_kwargs": {"wandb": {"config": vars(args)}}
        }


    # Initialize and run Trainer
    trainer = Pix2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dl=train_dl,
        val_dl=val_dl,
        optim=optim,
        scheduler=scheduler,
        accelerator_kwargs=accelerator_kwargs,
        **training_params,
        **logging_params,
     
    )

    trainer.train()