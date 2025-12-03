import torch
import torch.nn as nn
from dataset import CONFIG
'''
This file defines the LLaDAMs model architecture.
'''
class LLaDAMs(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG['d_model'])
        # Learnable Positional Embedding (Crucial for Diffusion)
        self.pos_emb = nn.Embedding(CONFIG['context_length'], CONFIG['d_model'])
        
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=CONFIG['d_model'], 
                nhead=CONFIG['n_heads'], 
                dim_feedforward=CONFIG['d_model']*4, 
                dropout=0.1, # Dropout helps prevent memorization
                batch_first=True, 
                norm_first=True
            ),
            num_layers=CONFIG['n_layers']
        )
        self.head = nn.Linear(CONFIG['d_model'], vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.emb(x) + self.pos_emb(positions)
        return self.head(self.tf(x))

# model = LLaDAMs().to(device)
# optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
