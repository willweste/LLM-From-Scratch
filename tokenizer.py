from typing import List
import re

class SimpleTokenizerV1:
    """
    This is a basic tokenizer class for encoding words to Integer IDs
    and decoding Integer IDs to words
    """

    # the constructor takes a dictionary
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    # method to encode our text
    def encode(self, text: str) -> List[int]:
        # text to IDs
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [word.strip() for word in preprocessed if word.strip()]
        # Need to use str_to_int
        encoded_text = [self.str_to_int[word] for word in preprocessed]
        return encoded_text

    # method to decode our text
    def decode(self, ids: List[int]) -> List[str]:
        # IDs to text
        text_from_ids = ' '.join([self.int_to_str[i] for i in ids])
        text_from_ids = re.sub(r'\s+([,.?!"()\'])', r'\1', text_from_ids)
        return text_from_ids




