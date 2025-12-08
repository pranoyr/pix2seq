import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
import os
import random
import cv2
import numpy as np

# Import your separate tokenizer
from tokenizer import Pix2SeqTokenizer

from torchvision.utils import draw_bounding_boxes, save_image


# ==============================================================================
# 2. Corrected Visualization Function
# ==============================================================================
def visualize_batch(root_path):
    print("Starting Visualization...")
    tokenizer = Pix2SeqTokenizer(num_bins=1000)
    collate_fn = Pix2SeqCollate(tokenizer)

    train_img = os.path.join(root_path, 'images', 'train2017')
    train_ann = os.path.join(root_path, 'annotations', 'instances_train2017.json')

    # Define Dataset
    dataset = MultiScaleCocoDataset(
        root=train_img,
        annFile=train_ann,
        sizes=(320, 352, 384, 416, 448, 480),
        max_size=640
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # --- 1. Get Batch (Handling Dictionary Return) ---
    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return

    # Extract from Dictionary
    images = batch["images"]
    tokens = batch["tokens"]
    valid_sizes = batch["valid_sizes"]

    print(f"Batch Padded Shape: {images.shape}")
    print(f"Valid Sizes (First 4): {valid_sizes.tolist()}")
    
    vis_images = []

    # --- 2. Draw ---
    for i in range(images.size(0)):
        
        # Retrieve the VALID (resized) dimensions for this specific image
        # This tells us the size of the "content" inside the padded tensor
        h_valid, w_valid = valid_sizes[i].tolist()
        
        # Decode tokens to Normalized coordinates [0, 1]
        boxes_norm, class_indices = tokenizer.decode(tokens[i])
        
        boxes_abs = []
        labels = []
        
        for box, cls_idx in zip(boxes_norm, class_indices):
            # box is [x, y, w, h] normalized relative to h_valid, w_valid
            
            # --- CORRECT SCALING ---
            # Multiply normalized coords by the VALID dimensions
            # (Because tokens were normalized by valid dimensions in collate)
            x = box[0] * w_valid
            y = box[1] * h_valid
            w = box[2] * w_valid
            h = box[3] * h_valid
            
            # Convert to [xmin, ymin, xmax, ymax]
            boxes_abs.append([x, y, x + w, y + h])
            labels.append(str(cls_idx))

        # Prepare image (uint8)
        img_uint8 = (images[i] * 255).to(torch.uint8)
        
        if len(boxes_abs) > 0:
            img_with_boxes = draw_bounding_boxes(
                img_uint8, 
                boxes=torch.tensor(boxes_abs), 
                labels=labels,
                colors="green", # Green for Ground Truth
                width=2
            )
            vis_images.append(img_with_boxes.float() / 255.0)
        else:
            vis_images.append(img_uint8.float() / 255.0)

    # --- 3. Save ---
    grid = torchvision.utils.make_grid(vis_images, nrow=2)
    save_image(grid, "correct_dataloader_viz.png")
    print("\n✅ Saved 'correct_dataloader_viz.png'. Check this file!")





# --- 1. Dataset Class (UPDATED WITH MAPPING) ---
class MultiScaleCocoDataset(CocoDetection):
    def __init__(self, root, annFile, sizes=(320, 352, 384, 416, 448, 480), max_size=640):
        super().__init__(root, annFile)
        self.sizes = sizes
        self.max_size = max_size
        
        # --- NEW MAPPING LOGIC START ---
        # COCO IDs are [1, ..., 90] with gaps. 
        # We map them to [0, ..., 79]
        self.valid_ids = sorted(self.coco.getCatIds())
        self.id_to_idx = {coco_id: i for i, coco_id in enumerate(self.valid_ids)}

        # --- NEW MAPPING LOGIC END ---

        self.base_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def get_target_size(self, h, w):
        short_side_target = random.choice(self.sizes)
        min_original_size = float(min(h, w))
        max_original_size = float(max(h, w))
        
        scale = short_side_target / min_original_size
        if max_original_size * scale > self.max_size:
            scale = self.max_size / max_original_size
            
        new_h = int(h * scale)
        new_w = int(w * scale)
        return (new_h, new_w), scale

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        w_orig, h_orig = img.size
        (new_h, new_w), scale = self.get_target_size(h_orig, w_orig)
        
        # Resize Image
        img_tensor = self.base_transform(img)
        img_tensor = v2.functional.resize(img_tensor, (new_h, new_w), antialias=True)

        # Resize Boxes
        boxes = []
        labels = []
        scale_w = new_w / w_orig
        scale_h = new_h / h_orig

        for obj in target:
            x, y, w, h = obj['bbox']
            new_box = [x * scale_w, y * scale_h, w * scale_w, h * scale_h]
            boxes.append(new_box)
            
            # --- UPDATED: USE MAPPING ---
            raw_id = obj['category_id']
            # Map 90 -> 79, etc.
            if raw_id in self.id_to_idx:
                labels.append(self.id_to_idx[raw_id])
            else:
                # Handle edge case if annotation has invalid ID
                continue 
            # -----------------------------

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target_dict = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
            "orig_size": torch.tensor([h_orig, w_orig]),
            "size": torch.tensor([new_h, new_w]) 
        }

        return img_tensor, target_dict
    

class Pix2SeqCollate:
    def __init__(self, tokenizer, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        images = [item[0] for item in batch]
        
        # 1. Pad Images (Your existing logic)
        max_h = max([img.shape[1] for img in images])
        max_w = max([img.shape[2] for img in images])
        padded_images = torch.zeros(len(images), 3, max_h, max_w, dtype=torch.float32)
        for i, img in enumerate(images):
            c, h, w = img.shape
            padded_images[i, :c, :h, :w] = img

        batch_tokens = []
        targets_dicts = [item[1] for item in batch]
        
        # We need to save the valid sizes for the Visualizer!
        valid_sizes = [] 

        for t in targets_dicts:
            boxes = t['boxes'] 
            labels = t['labels']
            
            # Get valid dimensions
            current_h, current_w = t['size'].tolist()
            valid_sizes.append([current_h, current_w]) # Store for later
            
            # ✅ CORRECT: Normalize relative to the valid image content
            seq = self.tokenizer.encode(boxes, labels, (current_h, current_w))
            batch_tokens.append(seq)

        # 2. Pad Tokens
        max_len = max([s.size(0) for s in batch_tokens])
        max_len = min(max_len, self.max_seq_len)
        padded_tokens = torch.full((len(batch), max_len), self.tokenizer.pad_id, dtype=torch.long)
        
        for i, seq in enumerate(batch_tokens):
            length = min(seq.size(0), max_len)
            padded_tokens[i, :length] = seq[:length]
        
        # Return 3 things: Images, Tokens, and Valid Sizes
        return {"images": padded_images, "tokens": padded_tokens, "valid_sizes": torch.tensor(valid_sizes, dtype=torch.int64)}
    


# --- 3. Loader Factory (Updated to return Tokenizer) ---
def get_pix2seq_dataloaders(root_path, batch_size=4, num_workers=2):
    
    # Init Tokenizer
    tokenizer = Pix2SeqTokenizer(num_bins=1000)
    collate_fn = Pix2SeqCollate(tokenizer)

    # Paths
    train_img = os.path.join(root_path, 'images', 'train2017')
    train_ann = os.path.join(root_path, 'annotations', 'instances_train2017.json')
    val_img = os.path.join(root_path, 'images', 'val2017')
    val_ann = os.path.join(root_path, 'annotations', 'instances_val2017.json')

    print("Initializing Training Set...")
    train_dataset = MultiScaleCocoDataset(
        root=train_img,
        annFile=train_ann,
        sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640),
        max_size=640
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("Initializing Validation Set...")
    val_dataset = MultiScaleCocoDataset(
        root=val_img,
        annFile=val_ann,
        sizes=(640,), 
        max_size=640
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


    # Return tokenizer so model can use the correct vocab_size
    return train_loader, val_loader

# --- 4. Main Execution ---
if __name__ == "__main__":

    
    COCO_ROOT = '/run/media/pranoy/Datasets/coco-dataset/coco/'


    visualize_batch(COCO_ROOT)

    if os.path.exists(COCO_ROOT):
        # Unpack the 3 return values
        tokenizer = Pix2SeqTokenizer(num_bins=1000)
        train_loader, val_loader = get_pix2seq_dataloaders(COCO_ROOT, batch_size=2, num_workers=0)
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Tokenizer Vocab Size: {tokenizer.vocab_size}")
        
        images, tokens = next(iter(train_loader))
        
        print(f"Images Shape: {images.shape}")
        print(f"Tokens Shape: {tokens.shape}")
        print(f"Max Token ID in batch: {tokens.max().item()}")




        
        # Validation Check
        # Ensure max token ID is within bounds
        assert tokens.max().item() < tokenizer.vocab_size, "CRITICAL: Token ID exceeds vocab size!"
        print("Validation Successful: All tokens are within vocabulary range.")
        
    else:
        print(f"Path not found: {COCO_ROOT}")