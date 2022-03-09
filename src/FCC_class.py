import torch.nn as nn
import torch.nn.functional as F


class FCC(nn.Module):

    # Architecture constructor
    def __init__(self, units):
        """
        Fully-connected model for Fashion-MNIST dataset. The architecture consists of two hidden layers
        with unspecified units.
        :param units: a list [u1,u2] where u1 is the # of units in the 1st hidden layer and u2 in the 2nd hidden layer
        """
        super(FCC, self).__init__()

        self.fLayer1 = nn.Linear(
            28 * 28,  # input image 28x28
            units[0],
            bias=True
        )

        self.fLayer2 = nn.Linear(
            units[0],
            units[1],
            bias=True
        )

        self.fLayer3 = nn.Linear(
            units[1],
            10,       # output class 10
            bias=True
        )

    # Forward function
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fLayer1(x))  # input layer to first hidden layer
        x = F.relu(self.fLayer2(x))  # first hidden layer to second hidden layer

        # output layer
        x = F.log_softmax(self.fLayer3(x), dim=1)
        return x
