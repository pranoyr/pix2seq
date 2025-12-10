import numpy as np
import torch

class Pix2SeqTokenizer:
    def __init__(self, num_bins=1000):
        """
        Args:
            num_bins: Resolution for coordinates (e.g., 1000 bins = 224px / 1000 is sub-pixel precision).
            class_names: List of strings for classes (e.g., COCO classes).
        """
        self.num_bins = num_bins

        class_names = coco_classes
        
        # 1. Define Special Tokens
        self.specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        # 2. Define Coordinate Tokens (e.g., "<bin_0>", "<bin_1>"...)
        # We reserve indices 4 to (4 + num_bins - 1) for coordinates
        self.coord_start_id = len(self.specials)
        self.coord_end_id = self.coord_start_id + num_bins
        
        # 3. Define Class Tokens
        # Classes start after the last coordinate bin
        self.class_start_id = self.coord_end_id
            
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # 4. Build Mappings
        self.stoi = {} # String to ID
        self.itos = {} # ID to String
        
        # Add Specials
        for i, token in enumerate(self.specials):
            self.stoi[token] = i
            self.itos[i] = token
            
        # Add Coordinates (Virtual construction to save memory)
        # We don't necessarily need to populate stoi with "bin_500" strings, 
        # we can calculate it mathematically, but for debugging:
        for i in range(num_bins):
            token = f"bin_{i}"
            idx = self.coord_start_id + i
            self.itos[idx] = token
            # self.stoi[token] = idx # Optional, rarely used directly
            
        # Add Classes
        for i, name in enumerate(class_names):
            idx = self.class_start_id + i
            self.stoi[name] = idx
            self.itos[idx] = name


        # print(self.stoi)
        # print("-----")
        # print(self.itos)
        # exit()

            
        self.vocab_size = len(self.itos)

    def quantize(self, x):
        """Converts float [0, 1] to integer bin [0, num_bins-1]"""
        # Clamp to ensure 0-1 range
        x = np.clip(x, 0, 1)
        # Scale and floor
        return int(x * (self.num_bins - 1))

    def dequantize(self, bin_idx):
        """Converts integer bin to float [0, 1]"""
        return bin_idx / (self.num_bins - 1)

    def encode(self, boxes, labels, orig_size):
        """
        Args:
            boxes: Tensor/List [N, 4] format (x, y, w, h) or (xmin, ymin, xmax, ymax)
            labels: Tensor/List [N] class indices
            orig_size: (height, width) of the original image
            
        Returns:
            List of token IDs: [<BOS>, ymin, xmin, ymax, xmax, class, ..., <EOS>]
        """
        h_img, w_img = orig_size
        
        token_sequence = [self.bos_id]
        
        for box, label in zip(boxes, labels):
            # 1. Normalize Boxes to [0, 1]
            # Assuming Input is COCO format [x, y, w, h]
            x, y, w, h = box
            
            # Convert to Corners [xmin, ymin, xmax, ymax]
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            
            # Normalize
            ymin_norm = ymin / h_img
            xmin_norm = xmin / w_img
            ymax_norm = ymax / h_img
            xmax_norm = xmax / w_img
            
            # 2. Quantize Coordinates -> Bin IDs -> Token IDs
            # Note: Pix2Seq usually uses ordering [ymin, xmin, ymax, xmax, class]
            coords = [ymin_norm, xmin_norm, ymax_norm, xmax_norm]
            
            for c in coords:
                bin_val = self.quantize(c)
                token_id = self.coord_start_id + bin_val
                token_sequence.append(token_id)
                
            # 3. Add Class Token
            # Labels are indices (0 to 80 or 90). Map to token ID.
            class_token_id = self.class_start_id + int(label)
            token_sequence.append(class_token_id)
            
        token_sequence.append(self.eos_id)
        
        return torch.tensor(token_sequence, dtype=torch.long)

    def decode(self, token_ids):
        """
        Converts a sequence of token IDs back to boxes and labels.
        Returns:
            boxes: [[x, y, w, h], ...] (Normalized 0-1)
            classes: [class_idx, ...]
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        boxes = []
        classes = []
        
        # Remove BOS/EOS/PAD
        clean_tokens = [t for t in token_ids if t not in [self.bos_id, self.eos_id, self.pad_id]]
        
        # Pix2Seq format is chunks of 5: [y, x, y, x, class]
        n = 5
        chunks = [clean_tokens[i:i + n] for i in range(0, len(clean_tokens), n)]
        
        for chunk in chunks:
            if len(chunk) < 5:
                continue # Incomplete detection
                
            # Extract bins and class
            y1_id, x1_id, y2_id, x2_id, class_id = chunk
            
            # Check if tokens are actually within coordinate range
            if not (self.coord_start_id <= y1_id < self.coord_end_id): continue

            c_idx = class_id - self.class_start_id

            if c_idx == (self.num_classes - 1):
                continue
            
            # Dequantize Coordinates
            y1 = self.dequantize(y1_id - self.coord_start_id)
            x1 = self.dequantize(x1_id - self.coord_start_id)
            y2 = self.dequantize(y2_id - self.coord_start_id)
            x2 = self.dequantize(x2_id - self.coord_start_id)
            
            # Convert Corners [ymin, xmin, ymax, xmax] -> COCO [x, y, w, h]
            w = x2 - x1
            h = y2 - y1
            
            boxes.append([x1, y1, w, h])
            
            classes.append(c_idx)
            
        return boxes, classes
    
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'background']


# --- Integration Example ---

if __name__ == "__main__":
    # Example COCO Classes (partial)
    
    print(len(coco_classes), "classes total.")
    
    # 1. Initialize
    tokenizer = Pix2SeqTokenizer(num_bins=1000)
    
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Coord Range: {tokenizer.coord_start_id} - {tokenizer.coord_end_id}")
    print(f"Class Range: {tokenizer.class_start_id} - {tokenizer.class_start_id + len(coco_classes)}")
    
    # 2. Fake Data (One image with 2 objects)
    # Box format: [x, y, w, h] absolute pixels
    # Image size: 640x480
    orig_size = (480, 640) 
    
    # Object 1: x=100, y=100, w=50, h=50, class=0 (person)
    # Object 2: x=200, y=200, w=100, h=100, class=2 (car)
    boxes = [[100, 100, 50, 50], [200, 200, 100, 100]]
    labels = [0, 2]
    
    # 3. Encode
    tokens = tokenizer.encode(boxes, labels, orig_size)
    print("\nEncoded Tokens:", tokens)
    
    # 4. Decode
    decoded_boxes, decoded_labels = tokenizer.decode(tokens)
    print("\nDecoded Boxes (Normalized):", decoded_boxes)
    print("Decoded Labels:", decoded_labels)

    # decoded unnormalized boxes for verification
    unnormalized_boxes = []
    for box in decoded_boxes:
        x, y, w, h = box
        unnormalized_boxes.append([x * orig_size[1], y * orig_size[0], w * orig_size[1], h * orig_size[0]])
    print("\nDecoded Boxes (Unnormalized):", unnormalized_boxes)
    
    # Verify values roughly match (100/640 = 0.156)
    print(f"Expected x1 normalized: {100/640:.3f}")