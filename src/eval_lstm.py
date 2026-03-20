
import torch
import evaluate
import numpy as np

def run_evaluation(model, data_loader, vocab, device):
    model.eval()
    rouge_metric = evaluate.load("rouge")
    
    total_loss = 0
    total_tokens = 0
    references = []
    predictions = []
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"], reduction='sum')
    pad_idx = vocab.stoi["<PAD>"]

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            lengths = batch['lengths']
            
            # PPL
            logits, _ = model(input_ids, lengths=lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            total_loss += loss.item()
            total_tokens += lengths.sum().item()
            
            # ROUGE (3/4 -> 1/4)
            for b in range(input_ids.size(0)):
                full_seq = [t for t in input_ids[b].tolist() if t != pad_idx]
                if len(full_seq) < 4: continue
                
                split_idx = int(len(full_seq) * 0.75)
                prefix = full_seq[:split_idx]
                target = full_seq[split_idx:]
                
                pred_tokens = model.generate(prefix, len(target), vocab, device)
                
                ref_text = " ".join([vocab.itos[t] for t in target if t > 3])
                pred_text = " ".join([vocab.itos[t] for t in pred_tokens if t > 3])
                
                if ref_text.strip() and pred_text.strip():
                    references.append(ref_text)
                    predictions.append(pred_text)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    if not references:
        return {'ppl': ppl, 'rouge1': 0.0, 'rouge2': 0.0}

    results = rouge_metric.compute(predictions=predictions, references=references)
    
    return {
        'ppl': ppl,
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2']
    }
