import torch
import torch.nn as nn
from transformers import AutoModel

import torch

import torch.nn as nn
from transformers import AutoModel

import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

import torch
import torch.nn as nn
from transformers import AutoModel
from tokenizer import Pix2SeqTokenizer
import math


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x, h=None, w=None):
        """
        x: [Batch, SeqLen, Dim]
        h, w: Grid dimensions (Height, Width) of the features
        """
        b, seq_len, dim = x.shape
        
        # Fallback to sqrt if dimensions not provided (only works for squares)
        if h is None or w is None:
            h = w = int(math.sqrt(seq_len))
            
        # Safety Check
        if h * w != seq_len:
            raise ValueError(f"Feature grid {h}x{w} ({h*w}) does not match sequence length {seq_len}!")
        
        # 1. Create Grid Mask
        mask = torch.ones((b, h, w), device=x.device)
        
        # 2. Cumulative Sum
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        
        # 3. Normalize
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 4. Generate Sine/Cosine
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        # 5. Concatenate and Flatten
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos.flatten(1, 2)




class Pix2SeqModel(nn.Module):
    def __init__(self, dino_model_name='facebook/dinov2-base'):
        super().__init__()
        
        # 1. Vision Encoder (DINOv2) - FROZEN
        self.dino = AutoModel.from_pretrained(dino_model_name)
        for param in self.dino.parameters():
            param.requires_grad = False
            



        tokenizer = Pix2SeqTokenizer(
            num_bins=1000,
        )
            
        dino_dim = self.dino.config.hidden_size 
        decoder_dim = 256
        vocab_size = tokenizer.vocab_size
        print(f"Vocab Size: {vocab_size}")
        max_seq_len = 1024 
        
        self.pad_token_id = tokenizer.pad_id


        self.pos_embed = PositionEmbeddingSine(num_pos_feats=decoder_dim // 2, normalize=True)
        
        # 2. Projector
        self.projector = nn.Linear(dino_dim, decoder_dim)
        
        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 4. Embeddings
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim)
        self.text_pos_embedding = nn.Embedding(max_seq_len, decoder_dim)
        
        # 5. Output Head
        self.output_head = nn.Linear(decoder_dim, vocab_size)


        # Store special tokens for generation
        self.pad_token_id = tokenizer.pad_id
        self.bos_token_id = tokenizer.bos_id
        self.eos_token_id = tokenizer.eos_id
        
        # 6. Loss Function (Internal)
        # ignore_index is crucial so we don't calculate loss on padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='none')
        
    def forward(self, image, target_tokens=None, loss_masks=None):
        """
        image: (Batch, 3, 224, 224)
        target_tokens: (Batch, Seq_Len) - Full Ground Truth Padded Sequence
                       e.g. [<BOS>, A, 1, <EOS>, <PAD>]
        
        Returns:
            (loss, logits) if target_tokens is provided
            (logits) if target_tokens is None (Inference mode)
        """

        B, C, H, W = image.shape
        
        # DINOv2 Base patch size is 14
        patch_size = 14
        h_feat = H // patch_size
        w_feat = W // patch_size
        
        # --- A. Vision Encoder ---
        with torch.no_grad():
            outputs = self.dino(image)
            visual_feats = outputs.last_hidden_state[:, 1:, :] 
            
        # --- B. Project ---
        memory = self.projector(visual_feats)

        # Generate positional embeddings based on the shape of memory
        # pos shape: [Batch, 256, 256]
        pos = self.pos_embed(memory, h=h_feat, w=w_feat)

        # Add to memory (broadcasts correctly)
        memory = memory + pos
        
        # 1. Slice Inputs and Targets
        # Input to Decoder: Remove the LAST token (We never predict from <EOS> or the last <PAD>)
        dec_input = target_tokens[:, :-1] 
        
        # Target for Loss: Remove the FIRST token (We never predict <BOS>)
        labels = target_tokens[:, 1:]
        
        # 2. Prepare Embeddings
        seq_len = dec_input.size(1)
        pos_ids = torch.arange(seq_len, device=target_tokens.device).unsqueeze(0)
        tgt = self.token_embedding(dec_input) + self.text_pos_embedding(pos_ids)
        
        # 3. Create Masks based on dec_input length
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(target_tokens.device)
        tgt_padding_mask = (dec_input == self.pad_token_id)
        
        # 4. Decode
        output = self.decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 5. Calculate Logits
        logits = self.output_head(output)
        
        # 6. Calculate Loss
        # Flatten inputs: (Batch * Seq_Len, Vocab_Size)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        mask_slice = loss_masks[:, 1:] 
        mask_flat = mask_slice.reshape(-1)
        
        # Multiply Loss by Weight (0.0 for Noise Coords, 1.0 for others)
        masked_loss = loss * mask_flat
        
        # Average over non-zero weights (avoid div by zero)
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-6)
        
        return loss, logits
            

    @torch.no_grad()
    def generate(self, image, max_new_tokens=100, top_p=0.4, temperature=1.0):
        """
        Runs Inference with Nucleus (Top-P) Sampling.
        This improves recall significantly over Greedy Search.
        """
        self.eval()
        device = image.device
        batch_size = image.size(0)

        # Grid Prep
        B, C, H, W = image.shape
        h_feat = H // 14
        w_feat = W // 14

        # Encode
        outputs = self.dino(image)
        visual_feats = outputs.last_hidden_state[:, 1:, :] 
        memory = self.projector(visual_feats)
        pos = self.pos_embed(memory, h=h_feat, w=w_feat)
        memory = memory + pos

        # Start Token
        generated_seq = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            seq_len = generated_seq.size(1)
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            
            tgt = self.token_embedding(generated_seq) + self.text_pos_embedding(pos_ids)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            
            # Predict next token from last output
            last_logits = self.output_head(output[:, -1, :]) 
            
            # --- TOP-P SAMPLING LOGIC ---
            # 1. Apply Temperature
            logits = last_logits / temperature
            
            # 2. Sort probabilities
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 3. Remove tokens with cumulative probability above the threshold (top_p)
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 4. Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # 5. Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            # -----------------------------

            generated_seq = torch.cat([generated_seq, next_token], dim=1)
            
        return generated_seq


if __name__ == "__main__":
    # Initialize
    model = Pix2SeqModel()

    