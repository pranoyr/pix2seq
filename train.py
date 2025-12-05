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

from dataloader import get_dataloaders




def resume_from_checkpoint(device, filename, model, optim, scheduler):
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        global_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
        
        logging.info(f"Resumed from checkpoint: {filename} at step {global_step}")

        return global_step


def save_ckpt(accelerator, model, optim, scheduler, global_step, filename):
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
    model = TableCoordinatePredictor()
    # Train loders
    train_dl, val_dl = get_dataloaders(
        root_path=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


    # training parameters
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
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
            for batch in train_dl:
                images, targets = batch["images"], batch["tokens"]
                images = images.to(device)
                targets = targets.to(device)


                print(images.shape, targets.shape)
                exit()
            
                # =========================
                # GENERATOR TRAINING STEP
                # =========================
                with accelerator.accumulate(model):
                    
                    optim.zero_grad(set_to_none=True)
                    
                    with accelerator.autocast():
                        loss = model(images, target_tokens=targets)[0]
                        
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
                    if not (global_step % args.ckpt_every) and accelerator.is_main_process:
                        save_ckpt(accelerator,
                                  model,
                                  optim,
                                  scheduler,
                                  global_step,
                                  os.path.join(args.ckpt_saved_dir, f'{wandb.run.name}-step-{global_step}.pth'))
                    
                    if not (global_step % args.sample_every):
                        sample_images(
                            model,
                            val_dl,
                            accelerator,
                            device,
                            global_step,
                            
                        )

                    if not (global_step % args.eval_every):
                        validate(
                            model,
                            val_dl,
                            accelerator,
                            device,
                            global_step
                        )
                    
                    # Prepare logging
                    log_dict = {
                        "loss": loss.item(),
                        "lr": optim.param_groups[0]['lr']
                    }


                    
                    accelerator.log(log_dict, step=global_step)
                    global_step += 1

    accelerator.end_training()        
    print("Train finished!")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()

    # project / dataset
    parser.add_argument('--project_name', type=str, default='Table-VLLM')
    parser.add_argument('--root', type=str, default='/run/media/pranoy/Datasets/coco-dataset/coco/',help="Path to dataset")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loader workers")

    # training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=1000, help="LR warmup steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], help="Mixed precision training mode")


    # logging / checkpointing
    parser.add_argument('--ckpt_every', type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument('--eval_every', type=int, default=2000, help="Evaluate every N steps")
    parser.add_argument('--sample_every', type=int, default=100, help="Sample and log reconstructions every N steps")
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
