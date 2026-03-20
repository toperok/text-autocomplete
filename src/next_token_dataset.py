import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Vocab:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = str(sentence).split()
            frequencies.update(tokens)

        word_to_add = self.vocab_size - len(self.itos)
        most_common = frequencies.most_common(word_to_add)

        for idx, (word, _) in enumerate(most_common, start=len(self.itos)):
            self.itos[idx] = word
            self.stoi[word] = idx
    
    def encode(self, text):
        tokens = str(text).split()
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]
    
    def __len__(self):
        return len(self.itos)
    
class TweetDataset(Dataset):
    def __init__(self, csv_path, vocab, max_seq_len, limit=None):
        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.iloc[:limit]

        self.texts = df['text'].astype(str).tolist()
        self.vocab = vocab
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx] 
        encoded = self.vocab.encode(text)
        
        encoded = encoded[:self.max_seq_len - 2]
        sequence = [self.vocab.stoi["<SOS>"]] + encoded + [self.vocab.stoi["<EOS>"]]
        
        return {'tokens': torch.tensor(sequence, dtype=torch.long)}

def collate_fn(batch, pad_idx=0):
    tokens_list = [item['tokens'] for item in batch]
    tokens_list.sort(key=lambda x: len(x), reverse=True)
    
    padded_texts = pad_sequence(tokens_list, batch_first=True, padding_value=pad_idx)
    
    masks = (padded_texts != pad_idx).long()

    input_ids = padded_texts[:, :-1]
    target_ids = padded_texts[:, 1:]

    input_lengths = masks[:, :-1].sum(dim=1)
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'lengths': input_lengths
    }

def get_dataloader(csv_path, vocab, config, shuffle=False, limit=None):
    dataset = TweetDataset(
        csv_path=csv_path,
        vocab=vocab,
        max_seq_len=config["data"]["max_seq_len"],
        limit=limit
    )

    return DataLoader(
        dataset=dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, vocab.stoi["<PAD>"])
    )
