import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.module):
    def __init__(self):
        super(MLPClassifier,self).__init__()
        # define our classifier
        self.layer_1 = nn.Linear(28*28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 512)
        self.layer_4 = nn.linear(512,1024)
        self.classify = nn.linear(1024, 9)

    def forward(self,x):
        # forward through the graph with relu activation.
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.classify(x))
        return F.log_softmax(x)

