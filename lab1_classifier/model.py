import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier,self).__init__()
        # define our fully connected classifier
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
