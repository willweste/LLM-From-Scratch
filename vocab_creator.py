import re


def create_vocab(text_file_path):
    with open(text_file_path) as file:
        raw_text = file.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [word.strip() for word in preprocessed if word.strip()]

    all_words = sorted(set(preprocessed))

    # Special context tokens here
    all_words.extend(['<|unk|>', '<|endoftext|>'])

    vocab = {token: integer for integer, token in enumerate(all_words)}

    return vocab
