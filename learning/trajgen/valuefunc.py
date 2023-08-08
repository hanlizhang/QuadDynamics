################################################################################
# Fengjun Yang, 2022
# This file contains the various value function parameterizations.
################################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class ValueFunc(ABC):
    ''' This is the abstract class of value functions. It is inherited by
        different specific parameterizations.
    '''

    @abstractmethod
    def learn(self, dataset):
        pass

    @abstractmethod
    def pred(self, x0, ref):
        pass



################################################################################
# Neural network parameterizations
################################################################################

class NNValueFunc(ValueFunc):
    ''' This class parameterizes the value function as a neural network
    '''

    def __init__(self):
        ''' Initialize a neural network '''
        self.network = None

    def learn(self, dataset, gamma, num_epoch=100, lr=0.01, batch_size=64, verbose=False, print_interval=10):
        ''' The general training loop for NN value functions '''
        # Define loss function, optimizer and lr scheduler
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        lossfn = nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_epoch)
        # Training loop
        for epoch in range(num_epoch):
            for x, _, r, x_ in dataloader:
                self.network.zero_grad()
                # Compute loss as the difference of prediction and target
                pred = self.network(x)
                target = r + gamma * self.network(x_)
                loss = lossfn(pred, target)
                # Update weights and learning rate
                loss.backward()
                optimizer.step()
                scheduler.step()
            # Print out the loss if in verbose mode
            if (verbose and epoch % print_interval == 0):
                print('Epoch: {} \t Training loss: {}'.format(epoch + 1, loss.item()))

    def eval(self):
        for param in self.network.parameters():
            param.requires_grad = False
        self.network.eval()

    def pred(self, x0, ref):
        ''' The general prediction for NN value functions '''
        # print(x0)
        # print(ref)
        # import ipdb;
        # ipdb.set_trace()
        d0 = torch.cat([x0, ref]).double()
        return self.network(d0.unsqueeze(0))[0]


class MLPValueFunc(NNValueFunc):
    ''' Value functions paramterized as a multi-layer-perceptron '''

    # def __init__(self, input_size, widths):
    # Constructor
    # Parameters:
    #    - widths:       list of Integers indicating the width of each layer
    # '''
    # self.network = MLP(input_size, widths)
    # self.network = mlp_t
    def __init__(self, net):
        self.network = net


class MLP(nn.Module):
    ''' Multi-layer perceptron '''

    def __init__(self, input_size, widths):
        ''' Constructor '''
        super().__init__()
        # First layer
        layers = [nn.Linear(input_size, widths[0]).double(), nn.GELU()]
        # Hidden layers
        for w1, w2 in zip(widths[0:-2], widths[1:-1]):
            layers += [nn.Linear(w1, w2).double(), nn.GELU()]
        # Last layer
        layers.append(nn.Linear(widths[-2], widths[-1]).double())
        self.model = nn.Sequential(*layers)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model.to(device)

    def forward(self, x):
        return self.model(x)


