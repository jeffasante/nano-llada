import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train(device, model, dataset, dataloader, CONFIG, optimizer, loss_fn, scheduler, epoch, n_epochs, writer, log_interval, save_interval):
    # The Training Loop
    print(f"🚀 Starting Main Training on {device}...")
    print("This will take a few minutes. Watch the loss drop below 2.0.")

    # We limit the loop for the demo, but in real life you'd run full epochs
    # Let's run 1000 batches per epoch to keep it snappy for your portfolio
    steps_per_epoch = 500 

    for epoch in range(CONFIG['epochs']):
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        data_iter = iter(dataloader)
        
        avg_loss = 0
        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                
            x, y, m = batch['input'].to(device), batch['target'].to(device), batch['mask'].to(device)
            
            logits = model(x)
            loss = nn.CrossEntropyLoss(reduction='none')(logits.view(-1, dataset.vocab_size), y.view(-1))
            loss = (loss.view(x.shape) * m).sum() / (m.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Average Loss: {avg_loss/steps_per_epoch:.4f}")
        
    
    # Save the trained model to your local folder
    torch.save(model.state_dict(), "llada_toy.pt")
    print("✅ Model saved as 'llada_toy.pt'")

    print("🎉 Training Complete!")

def continue_training(optimizer, model, dataloader, device, dataset, CONFIG):
    print("🔥 Resuming Training (Letting it cook longer)...")

    # 1. Bump the Learning Rate slightly (to push past 2.5)
    # We update the existing optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.001 # A bit more aggressive than 3e-4

    # 2. Train for 10 more epochs
    extra_epochs = 10
    steps_per_epoch = 500

    for epoch in range(extra_epochs):
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"Extra Epoch {epoch+1}")
        data_iter = iter(dataloader)
        
        avg_loss = 0
        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                
            x, y, m = batch['input'].to(device), batch['target'].to(device), batch['mask'].to(device)
            
            logits = model(x)
            loss = nn.CrossEntropyLoss(reduction='none')(logits.view(-1, dataset.vocab_size), y.view(-1))
            loss = (loss.view(x.shape) * m).sum() / (m.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print(f"Extra Epoch {epoch+1} Loss: {avg_loss/steps_per_epoch:.4f}")

    # --- CHECK RESULT ---
    print("\n--- NEW GENERATION ---")
    model.eval()
    curr = torch.full((1, CONFIG['context_length']), dataset.mask_token_id, device=device)
    steps = 20

    for step in range(steps):
        with torch.no_grad():
            logits = model(curr)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            pred_ids = dist.sample()
            confidence = torch.max(probs, dim=-1).values
            
            # Simple Confidence Masking
            is_masked = (curr == dataset.mask_token_id)
            masked_confidence = confidence.clone()
            masked_confidence[~is_masked] = -1.0 
            
            target_revealed = int(CONFIG['context_length'] * (step + 1) / steps)
            current_revealed = (~is_masked).sum().item()
            num_to_reveal = max(0, target_revealed - current_revealed)
            
            if num_to_reveal > 0:
                top_conf_indices = torch.argsort(masked_confidence, descending=True)
                indices_to_reveal = top_conf_indices[0, :num_to_reveal]
                curr[0, indices_to_reveal] = pred_ids[0, indices_to_reveal]

    final_ids = curr[0].cpu().numpy()
    print("".join([dataset.itos[c] for c in final_ids if c != dataset.mask_token_id]))

    # Save the resumed model
    torch.save(model.state_dict(), "llada_toy.pt")
    print("✅ Resumed model saved as 'llada_toy.pt'")





if __name__ == "__main__":
    from dataset import ShakespeareDataset, CONFIG
    from model import LLaDAMs
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import os

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize dataset and dataloader
    dataset = ShakespeareDataset(CONFIG['context_length'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Initialize model
    model = LLaDAMs(dataset.vocab_size).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    # Check if model exists to resume training
    model_path = "llada_toy.pt"
    if os.path.exists(model_path):
        print(f"📁 Found existing model at {model_path}. Loading and resuming training...")
        model.load_state_dict(torch.load(model_path))
        # Call continue_training to resume
        continue_training(optimizer, model, dataloader, device, dataset, CONFIG)
    else:
        print("🚀 No existing model found. Starting fresh training...")
        # Call the train function
        train(device, model, dataset, dataloader, CONFIG, optimizer, None, None, 0, CONFIG['epochs'], None, None, None)