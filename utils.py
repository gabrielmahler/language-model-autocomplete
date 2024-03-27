import re
from nltk.tokenize import RegexpTokenizer
import numpy as np
import random

CONTEXT_SIZE = 5

# def get_data(filepath):
#     with open(filepath, 'r') as f:
#         data = f.read()
#     data = data.replace('\n', ' ')
#     data = data.replace('.', '<STOP>')
#     data = re.sub(r'[^\w\s]', '', data)
#     data = re.sub(r'\d', '', data)
#     data = data.lower()

#     tokenizer = RegexpTokenizer(r'\w+')
#     data = tokenizer.tokenize(data)
#     unique_words = list(set(data))
#     word2idx = {w: i for i, w in enumerate(unique_words)}

#     next = []
#     prev = []

#     for i in range(len(data) - CONTEXT_SIZE):
#         next.append(data[i + CONTEXT_SIZE])
#         prev.append(data[i:i + CONTEXT_SIZE])
    
#     X = np.zeros((len(prev), CONTEXT_SIZE, len(unique_words)), dtype=bool)
#     Y = np.zeros((len(next), len(unique_words)), dtype=bool)
#     for i, ws in enumerate(prev):
#         for j, w in enumerate(ws):
#             X[i, j, word2idx[w]] = 1
#         Y[i, word2idx[next[i]]] = 1

#     return X, Y, unique_words, word2idx

def get_data(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    data = data.replace('\n', ' ')
    data = re.sub(r'(?![.!?])[^a-zA-Z\s\n]', '', data) # this should replace preprocessing??
    # data = data.replace('.', '<STOP>')
    # data = re.sub(r'[^\w\s]', '', data)
    # data = re.sub(r'\d', '', data)
    # data = data.lower()

    # tokenizer = RegexpTokenizer(r'\w+')
    # data = tokenizer.tokenize(data)
    # unique_words = list(set(data))
    # word2idx = {w: i for i, w in enumerate(unique_words)}
    
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', data)
    random.shuffle(sentences)
    train = sentences[:int(len(sentences) * 0.9)]
    test = sentences[int(len(sentences) * 0.9):]

    train_data = []
    test_data = []
    train_vocab = set()
    test_vocab = set()

    print("train data lenght: ", len(train), " test data lenght: ", len(test))

    for sentence in train:
        words = sentence_splitter(sentence)
        train_data += words
        train_vocab.update(words)
    for sentence in test:
        words = sentence_splitter(sentence)
        test_data += words
        test_vocab.update(words)
    
    all_vocab = set.union(train_vocab, test_vocab)
    word2idx = {w: i for i, w in enumerate(list(all_vocab))}

    train_data = list(map(lambda x: word2idx[x], train_data))
    test_data = list(map(lambda x: word2idx[x], test_data))

    print('Finished preprocessing data')
    print('Train data size: {}'.format(len(train_data)), 'Test data size: {}'.format(len(test_data)))

    return train_data, test_data, word2idx, all_vocab

# def preprocess(filepath): # not sure if even necessary
#     pattern = r'[^a-zA-Z\s]'
#     with open(filepath, 'r') as file:
#         text = file.read()
#     clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
#     with open('data/shakespeare.txt', 'w') as file:
#         file.write(clean_text)

def sentence_splitter(input):
    input = input.replace('.', '<STOP>')
    input = re.sub(r'[^\w\s]', '', input)
    input = re.sub(r'\d', '', input)
    input = input.lower()
    return input.split()








    