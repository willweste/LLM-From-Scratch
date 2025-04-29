import re
from tokenizer import SimpleTokenizerV1
from vocab_creator import create_vocab
from data_sampling import TextSummarizationDataset
'''''
with open('the-verdict.txt', "r", encoding="utf-8") as file:
    raw_text = file.read()
print("Total number of character:", len(raw_text))
# print(raw_text[:99])


#   below is a test of using re to split up words

text = "Hello, world. Is this-- a test?"
split_phrase = re.split(r'([,.:;?_!"()\']|--|\s)', text)
stripped_split_phrase = [word.strip() for word in split_phrase if word.split()]
print(stripped_split_phrase)

#   splitting all words now

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
mapped_words = {token: word for token, word in enumerate(all_words)}
for token, word in enumerate(all_words):
    if token == 51:
        break
    # print(word: {word} and token: {token}')

'''

# First create the vocab the encoder and decoder will use
vocab = create_vocab("the-verdict.txt")
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


tokenizer_object = SimpleTokenizerV1(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

encoded_text = tokenizer_object.encode(text)
print(encoded_text)

decoded_text = tokenizer_object.decode(encoded_text)
print(decoded_text)


# First create the vocab the encoder and decoder will use
vocab = create_vocab("the-verdict.txt")
print(f'This my vocab size: {len(vocab)}')
# Create a tokenizer object to use
tokenizer_object = SimpleTokenizerV1(vocab)

# Load in our text we want to use
with open("the-verdict.txt") as file:
    raw_text = file.read()

enc_text = tokenizer_object.encode(raw_text)
print(len(enc_text))
sample_enc = enc_text[:50]

