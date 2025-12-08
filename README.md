# Pix2Seq: Object Detection as a Language Modeling Task

Pix2Seq is a novel framework that casts object detection as a language modeling problem. Instead of standard detection heads (like classification and regression branches), Pix2Seq uses a Transformer-based decoder to generate a sequence of tokens that represent bounding boxes and class labels.

### Key Concept
The core idea is to translate an image into a sequence of "words," where each "word" corresponds to an object coordinate ($y_{min}, x_{min}, y_{max}, x_{max}$) or a class label. The model learns the syntax of this object description language.

---

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 1.10+
* 🤗 Transformers
* 🤗 Accelerate
* Torchvision
* COCO API (for dataset)

### Installation

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare the COCO Dataset:**
    * Download the COCO 2017 dataset (train and validation split).
    * Ensure your directory structure looks like this:
        ```
        /path/to/coco/
        ├── images/
        │   ├── train2017/
        │   └── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
        ```

---

## Training

This implementation uses 🤗 Accelerate for simple and efficient multi-GPU training.

### 1. Configure Accelerate (One-Time Setup)
Run this command and follow the prompts to configure your distributed training environment (e.g., number of GPUs, mixed precision).
```bash
accelerate config


accelerate launch train.py \
    --root /path/to/coco/ \
    --project_name "Pix2Seq" \
    --batch_size 4 \
    --num_epochs 10 \
    --lr 1e-4 \
    --eval_every 2000 \
    --sample_every 1000 \
    --ckpt_saved_dir "./ckpt"