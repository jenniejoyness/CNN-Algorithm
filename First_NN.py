import torch.nn as nn
first_fc_layer = 5880
second_fc_layer = 1000
third_fc_layer = 100
ZERO = 0
ONE = 1
num_classes = 30

class FirstNet(nn.Module):

    def init_first_layer(self):

        # creates a set of convolutional filters.
        # param 1 - num of input channel
        # param 2 - num of output channel
        # param 3 - kernel_size - filter size 5*5
        return nn.Sequential(nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=2), nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))

    def init_second_layer(self):
        # creates a set of convolutional filters.
        # param 1 - num of input channel
        # param 2 - num of output channel
        # param 3 - kernel_size - filter size 5*5
        return nn.Sequential(nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))

    def init_third_layer(self):
        # creates a set of convolutional filters.
        # param 1 - num of input channel
        # param 2 - num of output channel
        # param 3 - kernel_size - filter size 5*5
        return nn.Sequential( nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

    def __init__(self):

        super(FirstNet, self).__init__()
        self.layer1 = self.init_first_layer()
        self.layer2 = self.init_second_layer()
        self.layer3 = self.init_third_layer()
        # avoid overfitting
        self.dropOut = nn.Dropout()
        self.fc1 = nn.Linear(first_fc_layer, second_fc_layer)
        self.fc2 = nn.Linear(second_fc_layer, third_fc_layer)
        self.fc3 = nn.Linear(third_fc_layer, num_classes)

    def forward(self, x):

        first_output = self.layer1(x)
        next_output = self.layer2(first_output)
        next_output = self.layer3(next_output)
        next_output = next_output.reshape(next_output.size(ZERO), -ONE)
        next_output = self.dropOut(next_output)
        next_output = self.fc1(next_output)
        next_output = self.fc2(next_output)
        final_output = self.fc3(next_output)
        return final_output
