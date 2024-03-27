import torch
import utils
from model import AutocompleteModel
import numpy as np

class Teller():
    def __init__(self, model_name) -> None:
        # self.model = torch.load(model_name)
        self.checkpoint = torch.load(model_name)
        self.vocab_size = self.checkpoint['vocab_size']
        self.word2idx = self.checkpoint['word2idx']
        self.model = AutocompleteModel(vocab_size=self.vocab_size, word2idx=self.word2idx)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.reverse_vocab = {idx: word for word, idx in self.word2idx.items()}
    
    def predict_next_word(self, input, sample_n=10):
        inputs = utils.sentence_splitter(input)
        in_tensor = torch.tensor([[]], dtype=torch.int8)
        text = []
        for word in inputs:
            if word in self.word2idx:
                w = torch.tensor([[self.word2idx[word]]])
                in_tensor = torch.cat((in_tensor, w), dim=1)

        if in_tensor.shape[1] == 0:
            return ""
        
        with torch.no_grad():
            logits = self.model(in_tensor)
            logits = logits[-1, -1, :]
            top_n = torch.argsort(logits)[-sample_n:]
            n_logits = torch.exp(logits[top_n]) / torch.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n, p=n_logits.numpy())
            text.append(self.reverse_vocab[out_index])
        for word in text:
            print(input + ' ' + word)
        return text[0]


        # next_input = in_tensor.clone()

        # with torch.no_grad():
        #     for i in range(length):
        #         logits = model(next_input)
        #         logits = logits[0, 0, :]
        #         top_n = torch.argsort(logits)[-sample_n:]
        #         n_logits = torch.exp(logits[top_n]) / torch.exp(logits[top_n]).sum()
        #         out_index = np.random.choice(top_n, p=n_logits.numpy())

        #         text.append(self.reverse_vocab[out_index])
        #         next_input = torch.tensor([[out_index]])
        # print(text)
        
        
        # for word in inputs:
        #     if word not in self.word2idx:
        #         print("not in vocab")
        #     else:
        #         generate_sentence(self.model, word, 20, self.word2idx, 10)


    


def generate_sentence(model, word1, length, vocab, sample_n=10):
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    first_string = word1
    first_word_index = vocab[word1]
    next_input = torch.tensor([[first_word_index]])

    text = [first_string]

    with torch.no_grad():
        for i in range(length):
            logits = model(next_input)
            logits = logits[0, 0, :]
            top_n = torch.argsort(logits)[-sample_n:]
            n_logits = torch.exp(logits[top_n]) / torch.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n, p=n_logits.numpy())

            text.append(reverse_vocab[out_index])
            next_input = torch.tensor([[out_index]])

    print(" ".join(text))
    return text

        # for word1 in ''.split():
    #     if word1 not in word2idx:
    #         print(f"{word1} not in vocabulary")
    #     else:
    #         generate_sentence(model, word1, 20, word2idx, 10)


# model = Teller('shakespeare_model.pt')
# test = 'doth'
# a = model.predict_next_word(test, 1)
# print(a)
# pass