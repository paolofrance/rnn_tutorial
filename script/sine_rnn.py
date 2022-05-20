import torch
from torch import nn
import numpy as np


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,device):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        print(self.rnn)
        print(self.fc)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # hidden=hidden.to('cuda:0')
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden


def do_nn():

    text = ['hey how are you','good i am fine','have a nice day']
    chars = set(''.join(text))
    int2char = dict(enumerate(chars))
    char2int = {char: ind for ind, char in int2char.items()}

    maxlen = len(max(text, key=len))

    for i in range(len(text)):
        while len(text[i]) < maxlen:
            text[i] += ' '

    input_seq = []
    target_seq = []

    for i in range(len(text)):
        input_seq.append(text[i][:-1])

        target_seq.append(text[i][1:])

    for i in range(len(text)):
        input_seq[i] = [char2int[character] for character in input_seq[i]]
        target_seq[i] = [char2int[character] for character in target_seq[i]]

    dict_size = len(char2int)
    seq_len = maxlen - 1
    batch_size = len(text)

    input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)

    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.Tensor(target_seq)

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda:0")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # Instantiate the model with hyperparameters
    model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1,device=device)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model.to(device)

    # Define hyperparameters
    n_epochs = 100
    lr=0.01

    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Training Run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        output, hidden = model(input_seq)
        print(input_seq)
        print(output)
        print(hidden)

        loss = criterion(output, target_seq.view(-1).long())
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))


    # This function takes in the model and character as arguments and returns the next character prediction and hidden state
    def predict(model, character):
        # One-hot encoding our input to fit into the model
        character = np.array([[char2int[c] for c in character]])
        character = one_hot_encode(character, dict_size, character.shape[1], 1)
        character = torch.from_numpy(character)
        character = character.to(device)
        out, hidden = model(character)

        prob = nn.functional.softmax(out[-1], dim=0).data
        # Taking the class with the highest probability score from the output
        char_ind = torch.max(prob, dim=0)[1].item()

        return int2char[char_ind], hidden

    # This function takes the desired output length and input characters as arguments, returning the produced sentence
    def sample(model, out_len, start='hey'):
        model.eval() # eval mode
        start = start.lower()
        # First off, run through the starting characters
        chars = [ch for ch in start]
        size = out_len - len(chars)
        # Now pass in the previous characters and get a new one
        for ii in range(size):
            char, h = predict(model, chars)
            chars.append(char)

        return ''.join(chars)

    response = sample(model, 15, 'good')

    print(response)

if __name__ == '__main__':
    do_nn()