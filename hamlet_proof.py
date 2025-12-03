import torch
import torch.nn as nn
from model import LLaDAMs
from dataset import get_shakespeare_tokenizer

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hamlet text for overfitting
hamlet_text = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles, And by opposing end them? To die: to sleep; No more; and by a sleep to say we end The heart-ache and the thousand natural shocks That flesh is heir to, 'tis a consummation Devoutly to be wish'd. To die, to sleep; To sleep: perchance to dream: ay, there's the rub; For in that sleep of death what dreams may come When we have shuffled off this mortal coil, Must give us pause: there's the respect That makes calamity of so long life; For who would bear the whips and scorns of time, The oppressor's wrong, the proud man's contumely, The pangs of despised love, the law's delay, The insolence of office and the spurns That patient merit of the unworthy takes, When he himself might his quietus make With a bare bodkin? who would fardels bear, To grunt and sweat under a weary life, But that the dread of something after death, The undiscover'd country from whose bourn No traveller returns, puzzles the will And makes us rather bear those ills we have Than fly to others that we know not of? Thus conscience does make cowards of us all; And thus the native hue of resolution Is sicklied o'er with the pale cast of thought, And enterprises of great pith and moment With this regard their currents turn awry, And lose the name of action. Soft you now! The fair Ophelia! Nymph, in thy orisons Be all my sins remember'd."

# Get tokenizer
tokenizer = get_shakespeare_tokenizer()
vocab_size = tokenizer['vocab_size']
mask_token_id = tokenizer['mask_token_id']
itos = tokenizer['itos']
stoi = tokenizer['stoi']

# Filter Hamlet text to known chars
hamlet_ids = [stoi[c] for c in hamlet_text if c in stoi]
data = torch.tensor(hamlet_ids, dtype=torch.long)

# Load or initialize model
model = LLaDAMs(vocab_size).to(device)
# Uncomment to load trained model
# model.load_state_dict(torch.load("path/to/trained_model.pt"))

# --- FINAL CORRECTED GENERATOR (Handles Remainders) ---
print("\n--- RECONSTRUCTING HAMLET (Final Polish) ---")
model.eval()
curr = torch.full((1, len(data)), mask_token_id, device=device)

steps = 12  # Increased slightly for smoothness
total_len = len(data)

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

    # 3. LOCKING LOGIC (With Remainder Fix)
    is_masked = (curr == mask_token_id)
    masked_confidence = confidence.clone()
    masked_confidence[~is_masked] = -1.0  # Ignore already revealed

    # Calculate exactly how many we should have revealed by now
    target_revealed = int(total_len * (step + 1) / steps)
    current_revealed = (~is_masked).sum().item()
    num_to_reveal = target_revealed - current_revealed

    # Force reveal at least 1 if needed, and ensure we don't go negative
    num_to_reveal = max(0, num_to_reveal)

    if num_to_reveal > 0:
        top_conf_indices = torch.argsort(masked_confidence, descending=True)
        indices_to_reveal = top_conf_indices[0, :num_to_reveal]
        curr[0, indices_to_reveal] = pred_ids[0, indices_to_reveal]

print("-" * 60)
final_ids = curr[0].cpu().numpy()
full_text = "".join([itos[c] for c in final_ids if c != mask_token_id])
print(f"Final | \n{full_text}")
