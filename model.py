import torch
import torch.nn as nn
import numpy as np
from utils import get_data
from types import SimpleNamespace
import sys


class AutocompleteModel(nn.Module):

    def __init__(self, vocab_size, word2idx, rnn_size=128, embed_size=64):
        super().__init__()

        self.word2idx = word2idx
        # self.training = training

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        self.e = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.rnn_size, batch_first=True)
        self.dense1 = nn.Linear(self.rnn_size, self.vocab_size)

    def forward(self, inputs):
        # if if traininginputs not in self.word2idx:
        #     return ""
        em = self.e(inputs)
        o, _ = self.lstm(em)
        d = self.dense1(o)
        return d

    # def get_vocab(self):
    #     return self.vocab


# def generate_sentence(model, word1, length, vocab, sample_n=10):
#     reverse_vocab = {idx: word for word, idx in vocab.items()}
#     first_string = word1
#     first_word_index = vocab[word1]
#     next_input = torch.tensor([[first_word_index]])

#     text = [first_string]

#     with torch.no_grad():
#         for i in range(length):
#             logits = model(next_input)
#             logits = logits[0, 0, :]
#             top_n = torch.argsort(logits)[-sample_n:]
#             n_logits = torch.exp(logits[top_n]) / torch.exp(logits[top_n]).sum()
#             out_index = np.random.choice(top_n, p=n_logits.numpy())

#             text.append(reverse_vocab[out_index])
#             next_input = torch.tensor([[out_index]])

#     print(" ".join(text))


def get_text_model(vocab, word2idx):
    def perplexity(y_true, y_pred):
        ce = nn.CrossEntropyLoss()(y_pred, y_true)
        result = torch.exp(ce)
        return result

    model = AutocompleteModel(vocab_size=len(vocab), word2idx=word2idx)

    loss_metric = nn.CrossEntropyLoss()
    acc_metric = perplexity
    my_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    return SimpleNamespace(
        model=model,
        epochs=10,
        batch_size=65,
        optimizer=my_optimizer,
        loss=loss_metric,
        metric=acc_metric
    )

def make_data(data):
    train_data, test_data, word2idx, all_vocab = get_data(data)


    def process_trigram_data(data):
        window_sz = 25
        remainder = (len(data) - 1) % window_sz
        data = np.array(data)[:-remainder]

        X = data[:-1].reshape(-1, window_sz)
        Y = data[1:].reshape(-1, window_sz)

        return X, Y

    print("Processing trigram data")
    X0, Y0 = process_trigram_data(train_data)
    X1, Y1 = process_trigram_data(test_data)
    print("Done processing trigram data")

    return X0, Y0, X1, Y1, word2idx, all_vocab

def shuffle_data(x0, y0, x1, y1):
    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)
    y1 = np.array(y1)

    indices = np.arange(len(x0))
    np.random.shuffle(indices)

    x0 = x0[indices]
    y0 = y0[indices]

    indices = np.arange(len(x1))
    np.random.shuffle(indices)

    x1 = x1[indices]
    y1 = y1[indices]

    return torch.from_numpy(x0), torch.from_numpy(y0), torch.from_numpy(x1), torch.from_numpy(y1)


def main(data, model_name):
    # train_data, test_data, vocab = get_data("data/train.txt", "data/test.txt")
    # train_data, test_data, word2idx, all_vocab = get_data(data)


    # def process_trigram_data(data):
    #     window_sz = 25
    #     remainder = (len(data) - 1) % window_sz
    #     data = np.array(data)[:-remainder]

    #     X = data[:-1].reshape(-1, window_sz)
    #     Y = data[1:].reshape(-1, window_sz)

    #     return X, Y

    # X0, Y0 = process_trigram_data(train_data)
    # X1, Y1 = process_trigram_data(test_data)

    X0, Y0, X1, Y1, word2idx, all_vocab = make_data(data)

    args = get_text_model(all_vocab, word2idx)
    model = args.model
    optimizer = args.optimizer
    loss_metric = args.loss
    acc_metric = args.metric

    print("Starting training")

    for epoch in range(args.epochs):
        if epoch != 0:
            X0, Y0, X1, Y1 = shuffle_data(X0, Y0, X1, Y1)
        model.train()
        for i in range(0, len(X0), args.batch_size):
            inputs = torch.tensor(X0[i:i+args.batch_size], dtype=torch.long)
            targets = torch.tensor(Y0[i:i+args.batch_size], dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_metric(outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{len(X0)}, Loss: {loss.item():.4f}')

        print("Epoch done, model eval")
        model.eval()
        print("Model eval done, measuring accuracy")
        with torch.no_grad():
            inputs = torch.tensor(X1, dtype=torch.long)
            targets = torch.tensor(Y1, dtype=torch.long)
            outputs = model(inputs)
            acc = acc_metric(targets.view(-1), outputs.view(-1, model.vocab_size))
        print("Accuracy measured", acc)

        # with torch.no_grad():
        #     # inputs = torch.tensor(X1, dtype=torch.long)
        #     targets = torch.tensor(Y1, dtype=torch.long)
        #     batch_size = 32  # Set the desired batch size
        #     num_samples = len(X1)
        #     num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches

        #     all_predictions = []

        #     for i in range(num_batches):
        #         inputs = torch.tensor(X1[i:i+batch_size], dtype=torch.long)
        #         # targets = torch.tensor(Y0[i:i+batch_size], dtype=torch.long)
        #         print("batch", i, 'out of', num_batches)

        #         # Convert the batch to a tensor and add the batch dimension
        #         # batch_inputs = torch.tensor(batch_inputs, dtype=torch.long).unsqueeze(0)

        #         # Make predictions for the current batch
        #         predictions = model(inputs)

        #         # Append the predictions to the list of all predictions
        #         all_predictions.append(predictions)

        # # Concatenate the predictions from all batches along the batch dimension
        #     print('concat')
        # predictions = torch.cat(all_predictions, dim=1)
        # print('acc measuring')
        # acc = acc_metric(targets.view(-1), predictions.view(-1, model.vocab_size))


        print(f"Epoch {epoch + 1}/{args.epochs}, Perplexity: {acc:.4f}")

    # torch.save(model.state_dict(), f'{model_name}.pt')
    print('finished traiing')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'word2idx': model.word2idx
        }, f'{model_name}.pt')
    print('saved model')
    print('done')
    # for word1 in ''.split():
    #     if word1 not in word2idx:
    #         print(f"{word1} not in vocabulary")
    #     else:
    #         generate_sentence(model, word1, 20, word2idx, 10)


if __name__ == '__main__':
    source = sys.argv[1]
    model_name = sys.argv[2]
    # source = 'data/shakespeare_original.txt'
    # model_name = 'shakespeare_model'
    main(source, model_name)