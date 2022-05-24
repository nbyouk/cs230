"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as m


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

        # convolution layers
        self.conv1 = nn.Conv2d(params.in_channels, params.hidden_channels1, 5, padding=2)
        self.conv2 = nn.Conv2d(params.hidden_channels1, params.hidden_channels2, 3, padding=1)
        self.conv3 = nn.Conv2d(params.hidden_channels2, params.hidden_channels3, 3, padding=(0,1)) 
        self.conv4 = nn.Conv2d(params.hidden_channels3, params.hidden_channels4, 3, padding=1) 
        self.conv5 = nn.Conv2d(params.hidden_channels4, params.hidden_channels5, 3, padding=1) 

        # max pool layer
        self.maxpool = nn.MaxPool2d((2,1))

        # the fully connected layer transforms the output to give the final output layer
        self.fc1 = nn.Linear(29*33*params.hidden_channels5, 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of input spectrograms, batch_size x 1 x 936 x 33

        Returns:
            out: (Variable) dimension batch_size
        """
        s = self.conv1(s)   # batch_size x hidden_channels1 x 936 x 33
        s = self.maxpool(s) # batch_size x hidden_channels1 x 468 x 33
        s = self.conv2(s)   # batch_size x hidden_channels2 x 468 x 33
        s = self.maxpool(s) # batch_size x hidden_channels2 x 234 x 33
        s = self.conv3(s)   # batch_size x hidden_channels3 x 232 x 33
        s = self.maxpool(s) # batch_size x hidden_channels3 x 116 x 33
        s = self.conv4(s)   # batch_size x hidden_channels4 x 116 x 33
        s = self.maxpool(s) # batch_size x hidden_channels4 x 58  x 33
        s = self.conv5(s)   # batch_size x hidden_channels5 x 58  x 33
        s = self.maxpool(s) # batch_size x hidden_channels5 x 29  x 33

        s = F.relu(s)
        s = s.contiguous()

        # reshape the Variable before passing to hidden layer
        s = s.view(s.shape[0], -1)

        # apply the fully connected layers and obtain the output
        s = self.fc1(s)
        s = F.relu(s)
        s = self.fc2(s)

        return s 

def confusion_matrix(outputs, labels):
    labels = labels.ravel()
    outputs = outputs.ravel()
    return m.confusion_matrix(labels, outputs, labels=[1., 0.])

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) binary true labels: AFIB present = 1, not present = 0

    Returns: (float) accuracy in [0,1]
    """
    CM = confusion_matrix(outputs, labels)

    return float(CM[0, 0] + CM[1, 1]) / len(labels)

def precision(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) binary true labels: AFIB present = 1, not present = 0

    Returns: (float) precision in [0,1]
    """

    # compute confusion matrix
    CM = confusion_matrix(outputs, labels)

    return float(CM[0, 0]) / (CM[0, 0] + CM[1, 0] + 1e-10)

def recall(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) binary true labels: AFIB present = 1, not present = 0

    Returns: (float) accuracy in [0,1]
    """

    # compute confusion matrix
    CM = confusion_matrix(outputs, labels)

    return float(CM[0, 0]) / (CM[0, 0] + CM[0, 1] + 1e-10)

def f1(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) binary true labels: AFIB present = 1, not present = 0

    Returns: (float) accuracy in [0,1]
    """

    # compute precision and recall 
    p = precision(outputs, labels)
    r = recall(outputs, labels)
    return 2*p*r/(p + r + 1e-10)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
    # could add more metrics such as accuracy for each token type
}
