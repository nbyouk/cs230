"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        TODO: define layers here

        Args:
            params: (Params) contains layer dimensions 
        """
        super(Net, self).__init__()

        # layer
        self.conv1 = nn.Conv2d(params.in_channels, params.hidden_channels1, 3, padding=1)
        self.conv2 = nn.Conv2d(params.hidden_channels1, params.hidden_channels2, 3, padding=1)

        # max pool layer
        self.maxpool = nn.MaxPool2d((3,1))

        # batch norm layer
        self.batchnorm = nn.BatchNorm2d(params.hidden_channels2)

        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Linear((params.in_dim_1//9)*(params.in_dim_2)*params.hidden_channels2, 1)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of input spectrograms, batch_size x 1 x 936 x 33

        Returns:
            out: (Variable) dimension batch_size
        """
        s = self.conv1(s)
        s = self.maxpool(s)
        s = self.conv2(s)
        s = self.maxpool(s)
        s = self.batchnorm(s)
        s = F.relu(s)
        s = s.contiguous()

        # reshape the Variable before passing to hidden layer
        s = s.view(s.shape[0], -1)

        # apply the fully connected layer and obtain the output
        s = self.fc(s)

        return s 

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) binary true labels: AFIB present = 1, not present = 0

    Returns: (float) accuracy in [0,1]
    """

    # unroll labels and outputs
    labels = labels.ravel()
    outputs = outputs.ravel()

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return float(np.sum(outputs == labels))/len(labels)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
