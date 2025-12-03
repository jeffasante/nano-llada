
import torch
from model import LLaDAMs

#  VISUALIZER  
def visualize_clean(model, question, mask_token_id, itos, device, stoi):
    model.eval()
    print(f"\nUser: {question}")
    
    q_ids = [stoi[c] for c in question + "\n" if c in stoi]
    prompt_len = len(q_ids)
    
    # Canvas
    dummy_len = 30
    curr = torch.tensor(q_ids + [mask_token_id]*dummy_len, device=device).unsqueeze(0)
    
    # Pad
    if curr.shape[1] < 64:
        pad = torch.zeros(1, 64 - curr.shape[1], dtype=torch.long, device=device)
        curr = torch.cat([curr, pad], dim=1)
    
    print("-" * 50)
    print(f"{'Step':<5} | {'Thinking Process'}")
    print("-" * 50)

    for step in range(15):
        with torch.no_grad():
            logits = model(curr)
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1).values
            
            # Masking Logic (Preserve Prompt)
            is_masked = (curr == mask_token_id)
            masked_confidence = confidence.clone()
            masked_confidence[~is_masked] = -1.0 
            masked_confidence[:, :prompt_len] = -1.0
            
            # Reveal Schedule
            target_revealed = int(dummy_len * (step + 1) / 15)
            resp_slice = curr[0, prompt_len:prompt_len+dummy_len]
            current_revealed = (resp_slice != mask_token_id).sum().item()
            
            num_to_reveal = max(0, target_revealed - current_revealed)
            
            if num_to_reveal > 0:
                top_conf = torch.argsort(masked_confidence, descending=True)
                indices = top_conf[0, :num_to_reveal]
                curr[0, indices] = pred_ids[0, indices]
            
            # Print
            chars = curr[0].cpu().numpy()
            resp_chars = chars[prompt_len : prompt_len+dummy_len]
            txt = "".join(['.' if c == mask_token_id else itos[c] for c in resp_chars if c!=0])
            
            if "." in txt:
                print(f"{step:<5} | {txt.split('.')[0] + '.'}")
            else:
                print(f"{step:<5} | {txt}")

def load_model(model_path, vocab_size, device):
    """
    Load the trained LLaDAMs model from the given path.
    
    Args:
        model_path (str): Path to the saved model state_dict.
        vocab_size (int): Vocabulary size for the model.
        device: Torch device (e.g., 'cpu', 'cuda', 'mps').
    
    Returns:
        model: Loaded and evaluated model.
    """
    model = LLaDAMs(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {model_path} and ready for inference!")
    return model
