from tokenizer import SimpleTokenizerV1
from vocab_creator import create_vocab
from data_sampling import TextSummarizationDataset
from torch.utils.data import DataLoader


def create_dataloader(txt,
                      batch_size=4,
                      input_len=6,
                      target_len=2,
                      stride=1,
                      shuffle=True,
                      drop_last=True,
                      num_workers=0):
    vocab = create_vocab("the-verdict.txt")
    tokenizer = SimpleTokenizerV1(vocab)
    dataset = TextSummarizationDataset(txt, tokenizer, input_len, target_len, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
