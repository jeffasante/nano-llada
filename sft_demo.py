import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from model import LLaDAMs
from utils import visualize_clean
from dataset import CONFIG, get_shakespeare_tokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Get tokenizer from dataset module
tokenizer = get_shakespeare_tokenizer()
vocab_size = tokenizer['vocab_size']
mask_token_id = tokenizer['mask_token_id']
itos = tokenizer['itos']
stoi = tokenizer['stoi']

#  SFT DATASET (Prompt Preserving)  
qa_pairs = [
    ("Who art thou?", "I am a model."),
    ("Where is the king?", "The king is in the castle."),
    ("To be or not to be?", "That is the question.")
]

class SFTDataset(Dataset):
    def __init__(self, qa_pairs, context_length=64):
        self.data = []
        for q, a in qa_pairs:
            full_str = q + "\n" + a
            full_ids = [stoi[c] for c in full_str if c in stoi]
            # Pad to context length
            if len(full_ids) < context_length:
                full_ids = full_ids + [0] * (context_length - len(full_ids))
            self.data.append((torch.tensor(full_ids, dtype=torch.long), len(q)+1))

    def __len__(self): return 1000

    def __getitem__(self, idx):
        full_seq, prompt_len = self.data[idx % len(qa_pairs)]
        
        # Mask only the Response (Figure 2b)
        t = random.random()
        real_len = (full_seq != 0).sum()
        response_len = real_len - prompt_len
        num_mask = max(1, int(t * response_len))
        
        masked_input = full_seq.clone()
        # Find indices after the prompt
        response_indices = torch.arange(prompt_len, real_len)
        mask_indices = response_indices[torch.randperm(len(response_indices))[:num_mask]]
        
        masked_input[mask_indices] = mask_token_id
        loss_mask = torch.zeros_like(full_seq, dtype=torch.float)
        loss_mask[mask_indices] = 1.0
        
        return {"input": masked_input, "target": full_seq, "mask": loss_mask}

if __name__ == "__main__":
    
    # Load Model
    model = LLaDAMs(vocab_size).to(device)
    # Ideally load weights: 
    # model = model.load_state_dict(torch.load("llada_toy.pt"))
    
    # Fine-Tune
    dataset = SFTDataset(qa_pairs)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    print("Fine-tuning on Q&A...")
    for i in range(20):
        for batch in dataloader:
            x, y, m = batch['input'].to(device), batch['target'].to(device), batch['mask'].to(device)
            loss = nn.CrossEntropyLoss(reduction='none')(model(x).view(-1, vocab_size), y.view(-1))
            loss = (loss.view(x.shape) * m).sum() / (m.sum() + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    visualize_clean(model, "Where is the king?", mask_token_id, itos, device, stoi)
    