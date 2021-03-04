import torch.nn as nn


class FullyConectedAutoEncoder(nn.Module):
    def __init__(self):
        super(FullyConectedAutoEncoder, self).__init__()
        # define our fully connected auto-encoder here without using convolution layers
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 8, bias=False),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 784, bias=False),
            nn.ReLU(),
        )

    def encode(self, x):
        # implement the encoding pass
        return self.encoder(x.view(x.shape[0], -1))

    def decode(self, x, shape):
        # implement the decoding pass
        b, c, h, w = shape
        return self.decoder(x).view(b, c, h, w)

    def forward(self, x):
        # store our shapes for later use
        tensor_shape = x.shape
        # forward through the graph and return output tensor
        latent = self.encode(x)
        x = self.decode(latent, tensor_shape)
        return x, latent
