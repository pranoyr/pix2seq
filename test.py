import torch
from tqdm import tqdm
import os

# --- Import your custom modules ---
# Assuming your files are named model.py and dataloader.py
# If classes are in the same file, just ensure they are defined before running this.
from model import TableCoordinatePredictor 
from dataloader import get_loaders, MultiTableTokenizer

def load_checkpoint(model, checkpoint_path):
    """
    Loads weights from the checkpoint file.
    Handles 'module.' prefix if trained with Accelerator/DDP.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
    
    # Extract state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint # Sometimes saved directly
        
    # # Fix for 'module.' prefix (happens if trained with accelerator/DDP)
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if k.startswith('module.'):
    #         new_state_dict[k[7:]] = v # Remove 'module.'
    #     else:
    #         new_state_dict[k] = v
            
    model.load_state_dict(state_dict)
    print("Weights loaded successfully!")
    return model

def run_inference(checkpoint_path, batch_size=2):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    tokenizer = MultiTableTokenizer()

    # 2. Initialize Tokenizer & Model
    model = TableCoordinatePredictor()
    
    # 3. Load Weights
    if os.path.exists(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path)
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    model.to(device)
    model.eval() # CRITICAL: Sets dropout/batchnorm to eval mode

    # 4. Get Data Loader
    # We use the train_loader or val_loader depending on what you want to test
    train_loader, val_loader = get_loaders(batch_size=1)
    
    # Use val_loader if it exists, otherwise test on train_loader
    target_loader = val_loader if val_loader is not None else train_loader
    
    print(f"Starting inference on {len(target_loader)} batches...")
    
    # # 5. Inference Loop
    # total_samples = 0
    # exact_matches = 0
    
    with torch.no_grad(): # Disable gradient calculation for speed
        for batch_idx, (batch) in enumerate(target_loader):

            images = batch["images"]
            target_tokens = batch["tokens"]
            excel_files = batch["excel_paths"]
            sheet_names = batch["sheet_names"]
            
            images = images.to(device)
            target_tokens = target_tokens.to(device)

            # --- GENERATE PREDICTIONS ---
            # Using the .generate() method we added to the model class
            # This runs the Greedy Search loop (predicting one token at a time)
            generated_ids = model.generate(images, max_new_tokens=1024)
            
            # --- DECODE & COMPARE ---
            print(f"\n--- Batch {batch_idx} ---")
            
            for i in range(len(images)):
                # Decode Ground Truth
                # We interpret the target_tokens tensor back to string
                gt_str = tokenizer.decode(target_tokens[i])
                
                # Decode Prediction
                pred_str = tokenizer.decode(generated_ids[i])
                
        
                print(f"Sample {i}:")
                print(f"  Excel File: {excel_files[i]}")
                print(f"  Sheet Name: {sheet_names[i]}")
                print(f"  GT:   {gt_str}")
                print(f"  Pred: {pred_str}")
              
            
            # Optional: Stop after a few batches to not spam console
            # if batch_idx >= 5: 
            #     print("\nStopping early for demo...")
            #     break


if __name__ == "__main__":
    # Path to your saved checkpoint
    # usually saved as 'ckpt/dvae/Table-VLLM-step-XXXX.pth' based on your training arguments
    CKPT_PATH = "ckpt/dazzling-river-9-step-40000.pth" 
    
    run_inference(CKPT_PATH)