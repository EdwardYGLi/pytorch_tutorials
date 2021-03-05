import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # define our convolutional classifier
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),  # 1x28x28
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1x14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 1x14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1x7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 1x7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.classify = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10, bias=False),
            nn.LogSoftmax())

    def forward(self, x):
        x = self.features(x)
        return self.classify(x.view(x.shape[0], -1))
