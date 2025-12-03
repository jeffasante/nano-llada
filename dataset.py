import torch
from torch.utils.data import Dataset, DataLoader
import requests
import random

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # You can change to "cuda" if using an NVIDIA GPU

CONFIG = {
    'context_length': 128,  # Longer context to learn sentence structure
    'd_model': 256,         # Wider model (more capacity)
    'n_heads': 8,
    'n_layers': 6,          # Deeper model
    'batch_size': 64,
    'lr': 3e-4,             # Standard Transformer learning rate
    'epochs': 5             # On full dataset, 5 epochs is a good start
}

class ShakespeareDataset(Dataset):
    def __init__(self, context_length=128):
        self.context_length = context_length
        
        # Download full TinyShakespeare
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("Downloading Full Dataset...")
        text = requests.get(url).text
        
        # Char-level tokenizer
        self.chars = sorted(list(set(text)))
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = len(self.chars) + 1 # +1 for [MASK]
        self.mask_token_id = len(self.chars)
        
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        print(f"✅ Data Ready: {len(self.data)} characters.")

    def __len__(self):
        # We limit epoch size to speed up feedback loop
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        # Random chunk from the text
        chunk = self.data[idx : idx + self.context_length]
        
        # Diffusion Noise
        t = random.random()
        num_mask = int(t * self.context_length)
        
        masked_input = chunk.clone()
        mask_indices = torch.randperm(self.context_length)[:num_mask]
        masked_input[mask_indices] = self.mask_token_id
        
        loss_mask = torch.zeros_like(chunk, dtype=torch.float)
        loss_mask[mask_indices] = 1.0
        
        return {"input": masked_input, "target": chunk, "mask": loss_mask}
    
    def get_vocab_info(self):
        return {
            'vocab_size': self.vocab_size,
            'mask_token_id': self.mask_token_id,
            'itos': self.itos,
            'stoi': self.stoi
        }

def get_shakespeare_tokenizer():
    """
    Get the tokenizer components for Shakespeare dataset without instantiating the full dataset.
    Returns a dict with vocab_size, mask_token_id, itos, stoi.
    """
    # Download the text
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    
    # Char-level tokenizer
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars) + 1  # +1 for [MASK]
    mask_token_id = len(chars)
    
    return {
        'vocab_size': vocab_size,
        'mask_token_id': mask_token_id,
        'itos': itos,
        'stoi': stoi
    }

# dataset = ShakespeareDataset(CONFIG['context_length'])
# # Shuffle is CRITICAL for real training so it doesn't memorize order
# dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
# print("✅ DataLoader is ready.")
