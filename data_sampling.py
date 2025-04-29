import re
from tokenizer import SimpleTokenizerV1
from vocab_creator import create_vocab
import torch
from torch.utils.data import Dataset

'''
This data sampling method involves a sliding window technique for next word prediction.
'''


class TextSummarizationDataset(Dataset):
    def __init__(self, raw_text: str, tokenizer, input_len: int = 6, target_len: int = 2, stride=2):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        sentences = re.split(r'(?<=[.!?]) +', raw_text)

        self.max_input_len = 0
        self.max_target_len = 0

        for i in range(0, len(sentences) - input_len - target_len + 1, stride):
            input_text = " ".join(sentences[i:i + input_len])
            target_text = " ".join(sentences[i + input_len:i + input_len + target_len])

            input_encoded = tokenizer.encode(input_text)
            target_encoded = tokenizer.encode(target_text)

            self.max_input_len = max(self.max_input_len, len(input_encoded))
            self.max_target_len = max(self.max_target_len, len(target_encoded))

            self.input_ids.append(torch.tensor(input_encoded, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_encoded, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def get_max_seq_len(self):
        return max(self.max_input_len, self.max_target_len)
