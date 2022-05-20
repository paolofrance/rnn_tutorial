import numpy as np

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,device):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        print(self.rnn)
        print(self.fc)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out[:,-1,:])

        return out, hidden


def do_nn():

    inp = np.arange(1,21,1)

    data = torch.from_numpy(inp).float()

    INPUT_SIZE = 1
    SEQ_LENGTH = 5
    HIDDEN_SIZE = 20
    NUM_LAYERS = 1
    BATCH_SIZE = 4

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda:0")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    print(device)

    model = Model(input_size=INPUT_SIZE, output_size=2, hidden_dim=HIDDEN_SIZE, n_layers=NUM_LAYERS,device=device)
    model.to(device)

    inputs = data.view(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    inputs = inputs.to(device)

    tg = np.array([[6.,7.],[11.,12.],[16.,17.],[21.,22.]])
    targets = torch.tensor(tg).float()
    targets = targets.to(device)

    n_epochs = 1000
    lr=0.01

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch

        output, hidden = model(inputs)

        print(inputs.shape)
        print(targets.shape)
        print(output.shape)

        # input()

        loss = criterion(output, targets)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

    model.eval()

    # t = np.array([1,2,3,4,5])
    t = np.array([51,52,53,54,55])
    t = torch.tensor(t).float()
    print(t.shape)

    t = t.view(1,5,1)
    print(t.shape)

    t = t.to(device)

    out, hidden = model(t)
    print(out)





if __name__ == '__main__':
    do_nn()