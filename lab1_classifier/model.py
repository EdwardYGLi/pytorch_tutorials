import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier,self).__init__()
        # define our classifier
        self.layer_1 = nn.Linear(28*28, 1024)
        self.layer_2 = nn.Linear(1024,512)
        self.classify = nn.Linear(512, 10)

    def forward(self,x):
        # flatten image from [b,28,28] to [b,28*28]
        x = torch.flatten(x, start_dim=1)
        # forward through the graph with relu activation.
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.classify(x))
        return F.log_softmax(x)


class TestClassifier(nn.Module):
    def __init__(self):
        super(TestClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
