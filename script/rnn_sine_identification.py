import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


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


class SinDataset(Dataset):
    def __init__(self, t, length_h,length_p):
        self.data = np.sin(t)
        self.length_h = length_h
        self.length_p = length_p

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx<len(self.data)-(self.length_h+self.length_p):
            # return self.data[idx:idx+self.length], self.data[idx+1:idx+self.length+1]
            return self.data[idx:idx+self.length_h], self.data[idx+self.length_h:idx+self.length_h+self.length_p]
        else:
            idx=idx-self.length_h-self.length_p-1
            # return self.data[idx:idx+self.length], self.data[idx+1:idx+self.length+1]
            return self.data[idx:idx+self.length_h], self.data[idx+self.length_h:idx+self.length_h+self.length_p]


def do_nn():

    t = np.arange(0,10.0,0.01)

    BATCH_SIZE = 10
    SEQ_LENGTH = 20
    INPUT_SIZE = 1
    OUTPUT_SIZE = 10
    HIDDEN_SIZE = 50
    NUM_LAYERS = 3

    n_epochs = 1000
    lr=0.00001

    # create dataset
    dataset = SinDataset(t, SEQ_LENGTH,OUTPUT_SIZE)

    # divide dataset in train and test
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    print(len(dataset), len(train_set), len(test_set))

    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # for i, batch in enumerate(train):
    #     print(i, batch[0])
    #     print(batch[1])

    # input()

    # b= next(iter(train))
    # print(b[0])
    # print(b[1])


    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda:0")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # device = torch.device("cpu")
    print(device)

    model = Model(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_dim=HIDDEN_SIZE, n_layers=NUM_LAYERS,device=device)
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print_shapes = True

    for epoch in range(1, n_epochs + 1):
        for i, data in enumerate(train):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch

            inputs, targets = data[0].to(device), data[1].to(device)

            inputs = inputs.view(BATCH_SIZE,SEQ_LENGTH,1)
            targets = targets.view(BATCH_SIZE,OUTPUT_SIZE)

            inputs = inputs.float()
            targets = targets.float()

            output, hidden = model(inputs)

            if print_shapes:
                print("input shape: "+str(inputs.shape))
                print("hidden shape: "+str(hidden.shape))
                print("output shape: "+str(output.shape))
                print("target shape: "+str(targets.shape))
                # print(output)
                # print(targets)
                print_shapes = False

            loss = criterion(output, targets)
            loss.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly

        for i, data in enumerate(test):
            test_inputs, test_targets = data[0].to(device), data[1].to(device)

            test_inputs = test_inputs.view(BATCH_SIZE, SEQ_LENGTH, 1)
            test_targets = test_targets.view(BATCH_SIZE, OUTPUT_SIZE)

            test_inputs = test_inputs.float()
            test_targets = test_targets.float()

            test_output, test_hidden = model(test_inputs)

            if print_shapes:
                print("test_input shape: " + str(test_inputs.shape))
                print("test_hidden shape: " + str(test_hidden.shape))
                print("test_output shape: " + str(test_output.shape))
                print("test_target shape: " + str(test_targets.shape))
                # print(test_output)
                # print(test_targets)
                print_shapes = False

            test_loss = criterion(test_output,test_targets)

        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.6f}".format(loss.item()), end=' ')
            print(" -- Test Loss: {:.6f}".format(test_loss.item()),)



    model.eval()

    b = next(iter(test))
    test_inputs, test_targets = b[0].to(device), b[1].to(device)

    test_inputs = test_inputs.view(BATCH_SIZE, SEQ_LENGTH, 1)
    test_targets = test_targets.view(BATCH_SIZE, OUTPUT_SIZE)

    test_inputs = test_inputs.float()
    test_out,hid = model(test_inputs)

    print("qui")
    print(test_inputs[0])
    print(test_out[0])
    print(test_targets[0])
    print(len(test_inputs[0]))
    print(len(test_out[0]))

    t = np.arange(0,len(test_inputs[0])+len(test_out[0]),1)

    plt.plot(t[0:len(test_inputs[0])],test_inputs[0].cpu().detach().numpy())
    plt.plot(t[len(test_inputs[0]):len(t)],test_out[0].cpu().detach().numpy())
    plt.plot(t[len(test_inputs[0]):len(t)],test_targets[0].cpu().detach().numpy())
    plt.show()


if __name__ == '__main__':
    do_nn()