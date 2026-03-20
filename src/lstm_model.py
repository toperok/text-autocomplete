import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Set dropout to 0 if num_layers=1 to avoid PyTorch warning
        lstm_dropout = dropout if num_layers > 1 else 0

        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=lstm_dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, lengths=None, hidden=None):
        embeds = self.dropout(self.embedding(x))

        if lengths is not None:
            packed_embeds = pack_padded_sequence(
                embeds,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            packed_output, hidden = self.lstm(packed_embeds, hidden)
            out, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            out, hidden = self.lstm(embeds, hidden)

        logits = self.fc(self.dropout(out)) 
        
        return logits, hidden

    
    def generate(self, start_tokens, max_tokens, vocab, device):
        self.eval()

        input_ids = torch.tensor([start_tokens], dtype=torch.long).to(device)

        generated = []
        hidden = None

        with torch.no_grad():
            _, hidden = self.forward(input_ids, lengths=None, hidden=None)
            last_token = input_ids[:, -1:] 

            for _ in range(max_tokens):
                logits, hidden = self.forward(last_token, lengths=None, hidden=hidden)
                
                next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
                
                if next_token == vocab.stoi["<EOS>"]:
                    break
                    
                generated.append(next_token)
                last_token = torch.tensor([[next_token]]).to(device)

        return generated