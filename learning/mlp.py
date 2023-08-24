"""
SYNOPSIS
    Implementation of multilayer perceptron network using JAX libraries
DESCRIPTION

    Contains one module:
    a) MLP - defines the layers and depth of the multilayer perceptron - with the coeff to traj map
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.1
"""

from flax import linen as nn
import torch.nn as tnn
import torch.nn.functional as F


class MLP(nn.Module):
    num_hidden: list
    num_outputs: int

    def setup(self):
        self.linear = [
            nn.Dense(features=self.num_hidden[i]) for i in range(len(self.num_hidden))
        ]
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):
        for i in range(len(self.num_hidden)):
            # import pdb;
            # pdb.set_trace()
            x = self.linear[i](x)
            x = nn.elu(x)
        x = self.linear2(x)
        return x
        # return x ** 2


class MLP_torch(tnn.Module):
    def __init__(self, inp_size=502, num_hidden=[50, 40, 20]):
        super(MLP_torch, self).__init__()

        self.inp_size = inp_size
        self.num_hidden = num_hidden

        # Hidden layers
        self.hidden = tnn.ModuleList()
        self.hidden.append(tnn.Linear(inp_size, num_hidden[0]))
        for k in range(len(num_hidden) - 1):
            self.hidden.append(tnn.Linear(num_hidden[k], num_hidden[k + 1]))

        # Final output layer
        self.linear2 = tnn.Linear(num_hidden[-1], 1)

    def forward(self, x):
        x = x.float()
        for i in range(len(self.num_hidden)):
            # import ipdb;
            # ipdb.set_trace()
            x = self.hidden[i](x)
            x = F.elu(x)
        x = self.linear2(x)
        return x
