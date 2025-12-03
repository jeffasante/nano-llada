import torch
import torch.nn as nn
from model import LLaDAMs
from dataset import ShakespeareDataset, CONFIG, get_shakespeare_tokenizer
from utils import load_model
import os

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Get tokenizer from dataset module
tokenizer = get_shakespeare_tokenizer()
vocab_size = tokenizer['vocab_size']
mask_token_id = tokenizer['mask_token_id']
itos = tokenizer['itos']
stoi = tokenizer['stoi']

# Load or initialize model
model_path = "llada_toy.pt"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, vocab_size, device)
else:
    print("No trained model found. Initializing untrained model for demo...")
    model = LLaDAMs(vocab_size).to(device)

# GENERATION DEMO
print("\n--- GENERATING TEXT ---")
model.eval()
curr = torch.full((1, CONFIG['context_length']), mask_token_id, device=device)

steps = 20  # Number of diffusion steps

print(f"{'Step':<5} | Text")
print("-" * 60)

for step in range(steps):
    # 1. Visualize
    chars_curr = curr[0].cpu().numpy()
    text = "".join(['.' if c == mask_token_id else itos[c] for c in chars_curr])
    # Print clean single line
    print(f"{step:<5} | {text[:60]}...")  # Truncate for display

    # 2. Predict
    with torch.no_grad():
        logits = model(curr)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1).values

    # 3. LOCKING LOGIC
    is_masked = (curr == mask_token_id)
    masked_confidence = confidence.clone()
    masked_confidence[~is_masked] = -1.0  # Ignore already revealed

    # Reveal schedule
    target_revealed = int(CONFIG['context_length'] * (step + 1) / steps)
    current_revealed = (~is_masked).sum().item()
    num_to_reveal = max(0, target_revealed - current_revealed)

    if num_to_reveal > 0:
        top_conf_indices = torch.argsort(masked_confidence, descending=True)
        indices_to_reveal = top_conf_indices[0, :num_to_reveal]
        curr[0, indices_to_reveal] = pred_ids[0, indices_to_reveal]

print("-" * 60)
final_ids = curr[0].cpu().numpy()
full_text = "".join([itos[c] for c in final_ids if c != mask_token_id])
print(f"Final Generated Text | \n{full_text}")
