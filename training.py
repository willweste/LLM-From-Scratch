from dataloader_utils import create_dataloader
import torch

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

''''
dataloader = create_dataloader(
    raw_text, batch_size=1, input_len=6, target_len=2, stride=2, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)
'''

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from data_sampling import TextSummarizationDataset
from tokenizer import SimpleTokenizerV1
from vocab_creator import create_vocab

# Load text and create vocab
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab = create_vocab("the-verdict.txt")
tokenizer = SimpleTokenizerV1(vocab)

# Initialize dataset
dataset = TextSummarizationDataset(raw_text, tokenizer, input_len=6, target_len=2, stride=2)

# Maximum sequence length
max_sequence_length = dataset.get_max_seq_len()

# Embedding dimensions
embedding_dim = 256
vocab_size = len(vocab)

# Token embedding layer
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

# Positional embedding layer
pos_embedding_layer = torch.nn.Embedding(max_sequence_length, embedding_dim)


# Collate function for padding sequences
def collate_fn(batch):
    inputs, targets = zip(*batch)
    pad_token_id = tokenizer.str_to_int['<|endoftext|>']
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_token_id)
    return inputs_padded, targets_padded


# DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# Fetch one batch from DataLoader
inputs, targets = next(iter(dataloader))

# Generate token embeddings
token_embeddings = token_embedding_layer(inputs)

# Generate positional embeddings
positions = torch.arange(inputs.size(1)).unsqueeze(0).repeat(inputs.size(0), 1)
positional_embeddings = pos_embedding_layer(positions)

# Combine token and positional embeddings
input_embeddings = token_embeddings + positional_embeddings

print("Token IDs shape:", inputs.shape)
print("Token embeddings shape:", token_embeddings.shape)
print("Positional embeddings shape:", positional_embeddings.shape)
print("Combined embeddings shape:", input_embeddings.shape)
