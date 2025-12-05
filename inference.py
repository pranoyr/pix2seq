import torch
import numpy as np
import os
from PIL import Image
from model import TableCoordinatePredictor
from dataloader import MultiTableTokenizer
import pandas as pd
import dataframe_image as dfi

# ==========================================
# 1. VISUALIZATION (STRICT COPY FROM TRAINING)
# ==========================================
def convert_excel_to_image(excel_path, sheet_name, save_path):
    """
    Generates the image EXACTLY how the training data was generated.
    """
    try:
        # CRITICAL FIX 1: Use header=None
        # This aligns the rows perfectly with your training data (Row 0 is data, not header)
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        
        # CRITICAL FIX 2: Copy the exact styling from your dataset script
        border_style = '1px solid black'
        styled_df = (df.style
            # This matches your "generate_png_if_missing" function
            .background_gradient(cmap='Blues') 
            .set_properties(**{'border': border_style, 'color': 'black'}) 
            .set_table_styles([
                {'selector': 'th', 'props': [('border', border_style)]}, 
                {'selector': 'td', 'props': [('border', border_style)]}  
            ])
            .set_table_attributes('style="border-collapse: collapse;"')
        )

        # Export using Chrome (same as training)
        dfi.export(styled_df, save_path, max_rows=-1, max_cols=-1, table_conversion="chrome")
        return True
        
    except Exception as e:
        print(f"Failed to render {excel_path}: {e}")
        return False

# ==========================================
# 2. PREPROCESSING (STRICT COPY FROM DATALOADER)
# ==========================================
def smart_resize_inference(image, target_size=(896, 896)):
    w, h = image.size
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)
    
    # Padding Color: Gray (128) -> Matches your dataloader
    new_image = Image.new("RGB", target_size, (128, 128, 128))
    new_image.paste(image, (0, 0))
    
    return new_image

def preprocess_image(image_path, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Load
    image = Image.open(image_path).convert("RGB")
    
    # 2. Smart Resize (Gray Padding)
    image = smart_resize_inference(image, target_size=(896, 896))
    
    # 3. Normalize (Simple 0-1 scaling, matching your simple ToTensor logic)
    img_array = np.array(image) / 255.0
    
    # 4. Convert to Tensor (C, H, W)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)
    
    return img_tensor.unsqueeze(0).to(device)

# ==========================================
# 3. PREDICTION
# ==========================================
def predict(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init
    tokenizer = MultiTableTokenizer()
    model = TableCoordinatePredictor() 
    
    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Fix 'module.' prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()

    # Preprocess
    img_tensor = preprocess_image(image_path, device)
    
    # Generate
    print("Generating prediction...")
    with torch.no_grad():
        generated_ids = model.generate(img_tensor, max_new_tokens=100)
    
    return tokenizer.decode(generated_ids[0])

# ==========================================
# 4. RUN
# ==========================================
if __name__ == "__main__":
    
    # INPUTS
    XLSX_PATH = "/media/pranoy/Datasets/VEnron2/VEnron2/691/1_Metrics 0609.xlsx"
    SHEET = "East"
    
    # Use a temp path for the inference image
    TEMP_IMG = "data/inference_temp_corrected.png"
    CKPT = "ckpt/dazzling-river-9-step-40000.pth"
    
    print(f"Processing: {XLSX_PATH}")
    
    # 1. Convert (Using the EXACT training logic)
    success = convert_excel_to_image(XLSX_PATH, SHEET, TEMP_IMG)
    
    # 2. Predict
    if success:
        try:
            coords = predict(TEMP_IMG, CKPT)
            print("\n" + "="*30)
            print(f"PREDICTED: {coords}")
            print("="*30 + "\n")
        except Exception as e:
            print(f"Inference Error: {e}")