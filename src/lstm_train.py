import torch
import torch.nn as nn

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        lengths = batch['lengths']
        
        optimizer.zero_grad()
        logits, _ = model(input_ids, lengths=lengths)
        
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), 
            target_ids.reshape(-1)
        )
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)
