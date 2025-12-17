import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from tokenizer import Pix2SeqTokenizer
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
#  Rotary Embedding Components
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        # Calculate theta values
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        
        # Outer product to get frequencies
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # Concat [sin, cos] logic implicitly happens by repeating freqs
        # We store them as complex numbers or just pre-computed sin/cos usually
        # Here we use the standard "cat" format for simple application
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        # Return the pre-computed embeddings for the current sequence length
        seq_len = x.shape[1]
        return self.emb[:seq_len, :].to(x.device)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, freqs):
    """
    x: [batch, seq_len, num_heads, head_dim]
    freqs: [seq_len, head_dim]
    """
    # Reshape freqs for broadcasting: [1, seq_len, 1, head_dim]
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())



class RoPEDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, head_dim=64, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = head_dim
        self.inner_dim = nhead * head_dim 

        dim_feedforward = 4 * d_model
        
        # ==========================
        #  SELF ATTENTION 
        # ==========================
        self.sa_q = nn.Linear(d_model, self.inner_dim)
        self.sa_k = nn.Linear(d_model, self.inner_dim)
        self.sa_v = nn.Linear(d_model, self.inner_dim)
        self.sa_out = nn.Linear(self.inner_dim, d_model)
        
        # ==========================
        # CROSS ATTENTION 
        # ==========================
        self.ca_q = nn.Linear(d_model, self.inner_dim)
        self.ca_k = nn.Linear(d_model, self.inner_dim)
        self.ca_v = nn.Linear(d_model, self.inner_dim)
        self.ca_out = nn.Linear(self.inner_dim, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def attention(self, q, k, v, mask=None, is_causal=False):
        """Helper for standard scaled dot product attention"""
        B, L, _ = q.shape
        q = q.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(B, L, self.inner_dim)
        return out

    def forward(self, tgt, memory, rope_emb, tgt_mask=None):
        tgt_norm = self.norm1(tgt)
        
        q = self.sa_q(tgt_norm)
        k = self.sa_k(tgt_norm)
        v = self.sa_v(tgt_norm)
        
        # Apply RoPE
        B, L, _ = q.shape
        q_rope = q.view(B, L, self.nhead, self.head_dim)
        k_rope = k.view(B, L, self.nhead, self.head_dim)
        
        q_rope = apply_rotary_pos_emb(q_rope, rope_emb)
        k_rope = apply_rotary_pos_emb(k_rope, rope_emb)
        
        q = q_rope.view(B, L, self.inner_dim)
        k = k_rope.view(B, L, self.inner_dim)
        
        tgt2 = self.attention(q, k, v, mask=tgt_mask, is_causal=False)
        tgt2 = self.sa_out(tgt2)
        
        tgt = tgt + self.dropout1(tgt2)

        tgt_norm = self.norm2(tgt)
        
        # Query comes from normalized Decoder state
        q_cross = self.ca_q(tgt_norm) 
        
        # Key/Value come from Memory (Usually already normalized by Encoder)
        k_cross = self.ca_k(memory)  
        v_cross = self.ca_v(memory)
        
        # cross attention
        tgt2 = self.attention(q_cross, k_cross, v_cross, mask=None)
        tgt2 = self.ca_out(tgt2)
        
        # Residual
        tgt = tgt + self.dropout2(tgt2)

        tgt_norm = self.norm3(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        
        # Residual
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x, h=None, w=None):
        b, seq_len, dim = x.shape
        if h is None or w is None:
            h = w = int(math.sqrt(seq_len))
        if h * w != seq_len:
            raise ValueError(f"Feature grid {h}x{w} ({h*w}) does not match sequence length {seq_len}!")
        mask = torch.ones((b, h, w), device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos.flatten(1, 2)


class Pix2SeqModel(nn.Module):
    def __init__(self, 
                dino_model_name='facebook/dinov2-base',
                decoder_dim=256,
                num_decoder_layers=6,
                nhead=8,
                dim_head=64,
                max_seq_len=1024):
        
        super().__init__()
        
        # 1. Vision Encoder (DINOv2) - FROZEN
        self.dino = AutoModel.from_pretrained(dino_model_name)
        for param in self.dino.parameters():
            param.requires_grad = False
            
        tokenizer = Pix2SeqTokenizer(num_bins=1000)
            
        dino_dim = self.dino.config.hidden_size 
        vocab_size = tokenizer.vocab_size
        print(f"Vocab Size: {vocab_size}")
        
        # Special Tokens
        self.pad_token_id = tokenizer.pad_id
        self.bos_token_id = tokenizer.bos_id
        self.eos_token_id = tokenizer.eos_id

        # Positional Embeddings for Image Features (Additive Sine)
        self.pos_embed_image = PositionEmbeddingSine(num_pos_feats=decoder_dim // 2, normalize=True)
        
    
        # Decoder: Rotary Embeddings
        self.rope = RotaryEmbedding(dim=dim_head, max_seq_len=max_seq_len)
        
        # Projector
        self.projector = nn.Linear(dino_dim, decoder_dim)
        
        # Decoder Layers
        self.decoder_layers = nn.ModuleList([
            RoPEDecoderLayer(d_model=decoder_dim, nhead=nhead, head_dim=dim_head)
            for _ in range(num_decoder_layers)
        ])
        
        # Embeddings (Just Token Embedding, no absolute pos embedding needed)
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim)
        
        # Output Head
        self.output_head = nn.Linear(decoder_dim, vocab_size)

        self.final_norm = nn.LayerNorm(decoder_dim)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='none')

    def forward(self, image, target_tokens=None, loss_masks=None):
        B, C, H, W = image.shape
        h_feat, w_feat = H // 14, W // 14
        
        # --- Vision Encoder ---
        with torch.no_grad():
            outputs = self.dino(image)
            visual_feats = outputs.last_hidden_state[:, 1:, :] 
            
        memory = self.projector(visual_feats)

        # --- Add Sine Embeddings to Memory ---
        pos_img = self.pos_embed_image(memory, h=h_feat, w=w_feat)
        memory = memory + pos_img
        
        # --- Decoder Prep ---
        dec_input = target_tokens[:, :-1] 
        labels = target_tokens[:, 1:]
        
        # Embed
        tgt = self.token_embedding(dec_input)
        
        # Generate RoPE Frequencies for this sequence length
        rope_emb = self.rope(tgt)
        
        # Create Causal Mask
        seq_len = dec_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(target_tokens.device)
        
        # Run through Decoder Layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, rope_emb, tgt_mask=tgt_mask)

        tgt = self.final_norm(tgt)
        
        # Output Head
        logits = self.output_head(tgt)
        
        
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        mask_slice = loss_masks[:, 1:] 
        mask_flat = mask_slice.reshape(-1)
        masked_loss = loss * mask_flat
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-6)
        
        return loss, logits

    @torch.no_grad()
    def generate(self, image, max_new_tokens=100, top_p=0.4, temperature=1.0):
        self.eval()
        device = image.device
        B, C, H, W = image.shape
        h_feat, w_feat = H // 14, W // 14

        # Encode Image
        outputs = self.dino(image)
        visual_feats = outputs.last_hidden_state[:, 1:, :] 
        memory = self.projector(visual_feats)
        pos_img = self.pos_embed_image(memory, h=h_feat, w=w_feat)
        memory = memory + pos_img

        # Initialize Sequence
        generated_seq = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Embed
            tgt = self.token_embedding(generated_seq)
            
            # RoPE
            rope_emb = self.rope(tgt)
            
            # Mask
            seq_len = generated_seq.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            # Decoder Layers
            curr_tgt = tgt
            for layer in self.decoder_layers:
                curr_tgt = layer(curr_tgt, memory, rope_emb, tgt_mask=tgt_mask)

            curr_tgt = self.final_norm(curr_tgt)
            
            # Predict
            last_logits = self.output_head(curr_tgt[:, -1, :]) 
            
            # Sampling
            logits = last_logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            generated_seq = torch.cat([generated_seq, next_token], dim=1)
            
        return generated_seq

if __name__ == "__main__":
    # Initialize
    model = Pix2SeqModel()
    x = torch.randn(2, 3, 640, 640)
    tgt = torch.randint(0, 1000, (2, 50))
    loss, logits = model(x, tgt, loss_masks=torch.ones_like(tgt))
    print("Loss:", loss.item())
    print("Logits Shape:", logits.shape)

    