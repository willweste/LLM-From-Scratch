import re
from tokenizer import SimpleTokenizerV1
from vocab_creator import create_vocab

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
#print(vocab)
tokenizer_object = SimpleTokenizerV1(vocab)
text = "It's the last he painted, you know."

encoded_text = tokenizer_object.encode(text)
print(encoded_text)

decoded_text = tokenizer_object.decode(encoded_text)
print(decoded_text)