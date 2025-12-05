import torch
import torch.nn as nn
from transformers import AutoModel

import torch

import torch.nn as nn
from transformers import AutoModel

import torch.nn as nn
from transformers import AutoModel

import torch
import torch.nn as nn
from transformers import AutoModel



class Pix2SeqModel(nn.Module):
    def __init__(self, dino_model_name='facebook/dinov2-base'):
        super().__init__()
        
        # 1. Vision Encoder (DINOv2) - FROZEN
        self.dino = AutoModel.from_pretrained(dino_model_name)
        for param in self.dino.parameters():
            param.requires_grad = False
            
        dino_dim = self.dino.config.hidden_size 
        decoder_dim = 256
        vocab_size = len(tokenizer.stoi) 
        max_seq_len = 1024 
        
        self.pad_token_id = tokenizer.stoi['<PAD>']
        
        # 2. Projector
        self.projector = nn.Linear(dino_dim, decoder_dim)
        
        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # 4. Embeddings
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim)
        self.text_pos_embedding = nn.Embedding(max_seq_len, decoder_dim)
        
        # 5. Output Head
        self.output_head = nn.Linear(decoder_dim, vocab_size)


        # Store special tokens for generation
        self.pad_token_id = tokenizer.stoi['<PAD>']
        self.bos_token_id = tokenizer.stoi['<BOS>'] # Needed for starting generation
        self.eos_token_id = tokenizer.stoi['<EOS>'] # Needed for stopping
        
        # 6. Loss Function (Internal)
        # ignore_index is crucial so we don't calculate loss on padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        
    def forward(self, image, target_tokens=None):
        """
        image: (Batch, 3, 224, 224)
        target_tokens: (Batch, Seq_Len) - Full Ground Truth Padded Sequence
                       e.g. [<BOS>, A, 1, <EOS>, <PAD>]
        
        Returns:
            (loss, logits) if target_tokens is provided
            (logits) if target_tokens is None (Inference mode)
        """
        
        # --- A. Vision Encoder ---
        with torch.no_grad():
            outputs = self.dino(image)
            visual_feats = outputs.last_hidden_state[:, 1:, :] 
            
        # --- B. Project ---
        memory = self.projector(visual_feats)
        
        # --- C. Handle Training vs Inference ---
        if target_tokens is not None:
            # === TRAINING MODE ===
            
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
            
            return loss, logits
            
        else:
            # === INFERENCE MODE (Handled differently usually) ===
            # This block would be used if you call model(image) and generate manually
            return None, None # Placeholder


    @torch.no_grad()
    def generate(self, image, max_new_tokens=100):
        """
        Runs Greedy Search inference.
        Input: Batch of Images
        Output: Tensor of generated token IDs
        """
        self.eval()
        device = image.device
        batch_size = image.size(0)

        # 1. Encode Image ONCE (Memory)
        # We don't need to re-encode the image every loop
        outputs = self.dino(image)
        visual_feats = outputs.last_hidden_state[:, 1:, :] 
        memory = self.projector(visual_feats)

        # 2. Initialize Decoder Input with <BOS> token
        # Shape: (Batch_Size, 1) -> [[1], [1], [1]...]
        generated_seq = torch.full(
            (batch_size, 1), 
            self.bos_token_id, 
            dtype=torch.long, 
            device=device
        )

        # 3. Autoregressive Loop
        for _ in range(max_new_tokens):
            # Get current sequence length
            seq_len = generated_seq.size(1)

            # Create Position IDs
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # Embed inputs
            tgt = self.token_embedding(generated_seq) + self.text_pos_embedding(pos_ids)

            # Create Causal Mask (Standard procedure, though technically greedy only needs the last step)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            # Run Decoder
            # output shape: (Batch, Seq_Len, Dim)
            output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

            # Look at the LAST token output to predict the NEXT token
            last_token_output = output[:, -1, :]
            logits = self.output_head(last_token_output)

            # Greedy Pick: Take the token with the highest probability
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append prediction to the sequence
            generated_seq = torch.cat([generated_seq, next_token], dim=1)

            # Optimization: If valid, you can break here if ALL batches have generated <EOS>
            # But for simplicity, we just run until max_new_tokens
            
        return generated_seq


if __name__ == "__main__":
    # Initialize
    model = Pix2SeqModel()

    