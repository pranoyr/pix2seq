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


# --- 1. Dataset Class (Unchanged from your snippet) ---
class MultiScaleCocoDataset(CocoDetection):
    def __init__(self, root, annFile, sizes=(320, 352, 384, 416, 448, 480), max_size=640):
        super().__init__(root, annFile)
        self.sizes = sizes
        self.max_size = max_size
        
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
            labels.append(obj['category_id'])

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
            "size": torch.tensor([new_h, new_w]) # We need this for the tokenizer!
        }

        return img_tensor, target_dict

# --- 2. The Custom Collate Function ---
class Pix2SeqCollate:
    def __init__(self, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        # batch is a list of tuples: (img_tensor, target_dict)
        
        # A. Pad Images (Standard DETR/DINO approach)
        images = [item[0] for item in batch]
        max_h = max([img.shape[1] for img in images])
        max_w = max([img.shape[2] for img in images])
        
        padded_images = torch.zeros(len(images), 3, max_h, max_w, dtype=torch.float32)
        for i, img in enumerate(images):
            c, h, w = img.shape
            padded_images[i, :c, :h, :w] = img

        # B. Tokenize Targets
        batch_tokens = []
        targets_dicts = [item[1] for item in batch]
        
        for t in targets_dicts:
            boxes = t['boxes'] # These are RESIZED boxes
            labels = t['labels']
            # We must use the RESIZED dimensions for normalization to work
            # boxes are in pixels of 'size', not 'orig_size'
            current_h, current_w = t['size'].tolist()
            
            # Encode
            # Note: We assume COCO boxes [x, y, w, h] format
            seq = self.tokenizer.encode(boxes, labels, (current_h, current_w))
            batch_tokens.append(seq)

        # C. Pad Tokens
        # Find longest sequence in this batch
        max_len = max([s.size(0) for s in batch_tokens])
        max_len = min(max_len, self.max_seq_len)
        
        # Create padded tensor
        padded_tokens = torch.full((len(batch), max_len), self.tokenizer.pad_id, dtype=torch.long)
        
        for i, seq in enumerate(batch_tokens):
            length = min(seq.size(0), max_len)
            padded_tokens[i, :length] = seq[:length]
            
        return padded_images, padded_tokens

# --- 3. Loader Factory ---
def get_pix2seq_dataloaders(root_path, batch_size=4, num_workers=2):

    coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
    
    # Init Tokenizer
    # We create it here so we can pass it to the Collate function
    tokenizer = Pix2SeqTokenizer(num_bins=1000, class_names=coco_classes)
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
        collate_fn=collate_fn, # <--- Use our new collate
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
        collate_fn=collate_fn, # <--- Use our new collate
        pin_memory=True
    )

    return train_loader, val_loader

# --- 4. Main Execution ---
if __name__ == "__main__":
    COCO_ROOT = '/run/media/pranoy/Datasets/coco-dataset/coco/'

    if os.path.exists(COCO_ROOT):
        train_loader, val_loader = get_pix2seq_dataloaders(COCO_ROOT, batch_size=2, num_workers=0)
        
        print(f"\nTrain batches: {len(train_loader)}")
        
        # Fetch one batch to verify
        images, tokens = next(iter(train_loader))
        
        print(f"Images Shape: {images.shape} (Batch, C, H, W)")
        print(f"Tokens Shape: {tokens.shape} (Batch, Seq_Len)")


        print(tokens)
        
        # print("\n--- Sample 0 Analysis ---")
        # print(f"Token Sequence (First 20): {tokens[0][:20].tolist()}")
        
        # # Verify Token IDs
        # # 1 = BOS, 2 = EOS, 0 = PAD
        # # 4 to 1003 = Coordinates
        # # 1004+ = Classes
        # has_bos = (tokens[0] == 1).any().item()
        # has_eos = (tokens[0] == 2).any().item()
        # print(f"Has BOS? {has_bos}")
        # print(f"Has EOS? {has_eos}")
        
    else:
        print(f"Path not found: {COCO_ROOT}")