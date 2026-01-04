import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
import os
import random

# Import your separate tokenizer
from tokenizer import Pix2SeqTokenizer

from torchvision.utils import draw_bounding_boxes, save_image




def generate_noise_boxes(
    real_boxes, image_w, image_h, num_noise=3, perturbation_scale=0.2
):
    """
    Generates noise boxes using 50% Pure Random and 50% Perturbed Real strategies.
    Robust to small image sizes.
    """
    noise_boxes = []

    # --- SAFETY PRE-CALCULATION ---
    # Ensure the upper bound is at least 1 to prevent crashes on tiny images
    max_w_limit = max(1, image_w // 2)
    max_h_limit = max(1, image_h // 2)

    for _ in range(num_noise):
        # Strategy: 50% chance of Pure Random, 50% chance of Perturbed Real
        use_pure_random = (len(real_boxes) == 0) or (random.random() > 0.5)

        if use_pure_random:
            # === STRATEGY 1: Pure Random Box ===

            # FIX: Ensure lower bound is never larger than upper bound
            # If max_w_limit is 7 (image is 14px), we want randint(7, 7)
            # If max_w_limit is 100, we want randint(10, 100)
            low_w = min(10, max_w_limit)
            low_h = min(10, max_h_limit)

            w = random.randint(low_w, max_w_limit)
            h = random.randint(low_h, max_h_limit)

            x = random.randint(0, max(0, image_w - w))
            y = random.randint(0, max(0, image_h - h))
            noise_boxes.append([x, y, w, h])

        else:
            # === STRATEGY 2: Perturbed Real Box (Hard Negative) ===
            base_box = random.choice(real_boxes)  # [x, y, w, h]
            bx, by, bw, bh = base_box

            shift_x = int(bw * perturbation_scale * random.uniform(-1, 1))
            shift_y = int(bh * perturbation_scale * random.uniform(-1, 1))

            scale_w = random.uniform(1.0 - perturbation_scale, 1.0 + perturbation_scale)
            scale_h = random.uniform(1.0 - perturbation_scale, 1.0 + perturbation_scale)

            nw = int(bw * scale_w)
            nh = int(bh * scale_h)
            nx = int(bx + shift_x)
            ny = int(by + shift_y)

            # Clip to Image Boundaries
            nx = max(0, min(nx, image_w - 1))
            ny = max(0, min(ny, image_h - 1))
            nw = max(1, min(nw, image_w - nx))
            nh = max(1, min(nh, image_h - ny))

            noise_boxes.append([nx, ny, nw, nh])

    return noise_boxes


def visualize_batch(root_path):
    print("Starting Visualization...")
    tokenizer = Pix2SeqTokenizer(num_bins=1000)
    collate_fn = Pix2SeqCollate(tokenizer)

    train_img = os.path.join(root_path, "images", "train2017")
    train_ann = os.path.join(root_path, "annotations", "instances_train2017.json")

    # Define Dataset
    dataset = MultiScaleCocoDataset(root=train_img, annFile=train_ann, max_size=1024)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    images = batch["images"]
    tokens = batch["tokens"]
    valid_sizes = batch["valid_sizes"]

    print(f"Batch Padded Shape: {images.shape}")

    # --- 1. Define Inverse Normalization Constants ---
    # These must match the values in your Dataset __init__
    inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    vis_images = []

    for i in range(images.size(0)):
        h_valid, w_valid = valid_sizes[i].tolist()

        # --- 2. De-normalize the Image ---
        img_tensor = images[i].clone()  # Clone to keep original batch intact

        # Formula: original = (normalized * std) + mean
        img_tensor = img_tensor * inv_std + inv_mean

        # Clamp to ensure values don't go below 0 or above 1 due to float precision
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Now convert to uint8
        img_uint8 = (img_tensor * 255).to(torch.uint8)

        # --- 3. Decode Tokens (Same as before) ---
        boxes_norm, class_indices = tokenizer.decode(tokens[i])
        boxes_abs = []
        labels = []

        for box, cls_idx in zip(boxes_norm, class_indices):
            x = box[0] * w_valid
            y = box[1] * h_valid
            w = box[2] * w_valid
            h = box[3] * h_valid

            boxes_abs.append([x, y, x + w, y + h])
            labels.append(str(cls_idx))

        if len(boxes_abs) > 0:
            img_with_boxes = draw_bounding_boxes(
                img_uint8,
                boxes=torch.tensor(boxes_abs),
                labels=labels,
                colors="green",
                width=2,
            )
            vis_images.append(img_with_boxes.float() / 255.0)
        else:
            vis_images.append(img_uint8.float() / 255.0)

    grid = torchvision.utils.make_grid(vis_images, nrow=2)
    save_image(grid, "correct_dataloader_viz.png")
    print("\n✅ Saved 'correct_dataloader_viz.png'. Check this file!")


class MultiScaleCocoDataset(CocoDetection):
    def __init__(self, root, annFile, max_size=1024, is_train=True):
        super().__init__(root, annFile)

        self.max_size = max_size
        self.is_train = is_train

        # --- NEW MAPPING LOGIC START ---
        # COCO IDs are [1, ..., 90] with gaps.
        # We map them to [0, ..., 79]
        self.valid_ids = sorted(self.coco.getCatIds())
        self.id_to_idx = {coco_id: i for i, coco_id in enumerate(self.valid_ids)}

        self.noise_class_id = len(self.valid_ids)

        self.base_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_target_size(self, h, w):
        if self.is_train:
            # === TRAINING: Random Jitter ===
            scale = random.uniform(0.3, 2.0)

            # Clamp to max_size
            h_scaled = h * scale
            w_scaled = w * scale
            if max(h_scaled, w_scaled) > self.max_size:
                scale = self.max_size / max(h, w)

            new_h = int(h * scale)
            new_w = int(w * scale)

        else:
            # === VALIDATION: Deterministic Resize ===
            # Simply resize so the longest side equals max_size
            scale = self.max_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)

        # === DINOv2 FIX (Apply to BOTH Train and Val) ===
        patch_size = 14
        new_h = int(round(new_h / patch_size) * patch_size)
        new_w = int(round(new_w / patch_size) * patch_size)

        # Safety
        new_h = max(new_h, patch_size)
        new_w = max(new_w, patch_size)

        return (new_h, new_w)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        w_orig, h_orig = img.size
        (new_h, new_w) = self.get_target_size(h_orig, w_orig)

        # Resize Image
        img_tensor = self.base_transform(img)
        img_tensor = v2.functional.resize(img_tensor, (new_h, new_w), antialias=True)

        # Resize Boxes
        boxes = []
        labels = []
        scale_w = new_w / w_orig
        scale_h = new_h / h_orig

        for obj in target:
            x, y, w, h = obj["bbox"]
            new_box = [x * scale_w, y * scale_h, w * scale_w, h * scale_h]
            boxes.append(new_box)

            raw_id = obj["category_id"]
            # Map 90 -> 79, etc.
            if raw_id in self.id_to_idx:
                labels.append(self.id_to_idx[raw_id])
            else:
                # Handle edge case if annotation has invalid ID
                continue

        MAX_OBJECTS = 30

        num_real = len(boxes)
        num_noise = MAX_OBJECTS - num_real

        if num_noise < 0:
            num_noise = 0

        noise_boxes = generate_noise_boxes(
            boxes,
            image_w=new_w,  # Use the RESIZED dimensions
            image_h=new_h,
            num_noise=num_noise,
        )

        noise_labels = [self.noise_class_id] * len(noise_boxes)

        all_boxes = boxes + noise_boxes
        all_labels = labels + noise_labels

        # Weights: 1.0 for Real coordinates, 0.0 for Noise coordinates
        all_weights = [1.0] * len(boxes) + [0.0] * len(noise_boxes)

        combined = list(zip(all_boxes, all_labels, all_weights))
        random.shuffle(combined)
        all_boxes, all_labels, all_weights = zip(*combined)

        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(all_labels, dtype=torch.int64)
        weights_tensor = torch.tensor(all_weights, dtype=torch.float32)

        target_dict = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "weights": weights_tensor,
            "image_id": torch.tensor([index]),
            "orig_size": torch.tensor([h_orig, w_orig]),
            "size": torch.tensor([new_h, new_w]),
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
        batch_masks = []

        # We need to save the valid sizes for the Visualizer!
        valid_sizes = []

        for t in targets_dicts:
            boxes = t["boxes"]
            labels = t["labels"]
            weights = t["weights"]

            # Get valid dimensions
            current_h, current_w = t["size"].tolist()
            valid_sizes.append([current_h, current_w])  # Store for later

            # Normalize relative to the valid image content
            seq = self.tokenizer.encode(boxes, labels, (current_h, current_w))
            mask_list = [1.0]  # BOS is always valid

            # Iterate through the weights provided by Dataset
            for w in weights:
                val = w.item()  # 1.0 or 0.0

                # Append masks for 4 Coordinates + 1 Class
                # Coords get 'val' (0.0 if noise), Class ALWAYS gets 1.0
                mask_list.extend([val, val, val, val, 1.0])

            mask_list.append(1.0)  # EOS is always valid

            batch_tokens.append(seq)
            batch_masks.append(torch.tensor(mask_list, dtype=torch.float32))

        # 2. Pad Tokens AND Masks
        max_len = max([s.size(0) for s in batch_tokens])
        max_len = min(max_len, self.max_seq_len)

        padded_tokens = torch.full(
            (len(batch), max_len), self.tokenizer.pad_id, dtype=torch.long
        )
        padded_masks = torch.zeros(
            (len(batch), max_len), dtype=torch.float32
        )  # Pad with 0.0 (Ignore)

        for i, (seq, mask) in enumerate(zip(batch_tokens, batch_masks)):
            length = min(seq.size(0), max_len)
            padded_tokens[i, :length] = seq[:length]
            padded_masks[i, :length] = mask[:length]

        return {
            "images": padded_images,
            "tokens": padded_tokens,
            "loss_masks": padded_masks,
            "valid_sizes": torch.tensor(valid_sizes, dtype=torch.int64),
        }


# --- 3. Loader Factory (Updated to return Tokenizer) ---
def get_pix2seq_dataloaders(root_path, batch_size=4, num_workers=2):
    # Init Tokenizer
    tokenizer = Pix2SeqTokenizer(num_bins=1000)
    collate_fn = Pix2SeqCollate(tokenizer)

    # Paths
    train_img = os.path.join(root_path, "images", "train2017")
    train_ann = os.path.join(root_path, "annotations", "instances_train2017.json")
    val_img = os.path.join(root_path, "images", "val2017")
    val_ann = os.path.join(root_path, "annotations", "instances_val2017.json")

    print("Initializing Training Set...")
    train_dataset = MultiScaleCocoDataset(
        root=train_img, annFile=train_ann, max_size=1024, is_train=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print("Initializing Validation Set...")
    val_dataset = MultiScaleCocoDataset(
        root=val_img, annFile=val_ann, max_size=1024, is_train=False
    )

    # Create a fixed subset of the first 200 images
    mini_indices = list(range(200)) 
    val_dataset = torch.utils.data.Subset(val_dataset, mini_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Return tokenizer so model can use the correct vocab_size
    return train_loader, val_loader


# --- 4. Main Execution ---
if __name__ == "__main__":
    COCO_ROOT = "/run/media/pranoy/Datasets/coco-dataset/coco/"

    visualize_batch(COCO_ROOT)

    # if os.path.exists(COCO_ROOT):
    #     # Unpack the 3 return values
    #     tokenizer = Pix2SeqTokenizer(num_bins=1000)
    #     train_loader, val_loader = get_pix2seq_dataloaders(COCO_ROOT, batch_size=2, num_workers=0)

    #     print(f"\nTrain batches: {len(train_loader)}")
    #     print(f"Tokenizer Vocab Size: {tokenizer.vocab_size}")

    #     images, tokens = next(iter(train_loader))

    #     print(f"Images Shape: {images.shape}")
    #     print(f"Tokens Shape: {tokens.shape}")
    #     print(f"Max Token ID in batch: {tokens.max().item()}")

    #     # Validation Check
    #     # Ensure max token ID is within bounds
    #     assert tokens.max().item() < tokenizer.vocab_size, "CRITICAL: Token ID exceeds vocab size!"
    #     print("Validation Successful: All tokens are within vocabulary range.")

    # else:
    #     print(f"Path not found: {COCO_ROOT}")
