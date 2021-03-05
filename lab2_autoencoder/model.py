import torch.nn as nn


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, code_size=10):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.code_size = code_size
        # define our convolutional auto-encoder here without using convolution layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.latent_shape = [64, 7, 7]
        self.latent_enc = nn.Linear(7 * 7 * 64, self.code_size)

        self.latent_dec = nn.Linear(self.code_size, 7 * 7 * 64)

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def encode(self, x):
        # implement the encoding pass
        x = self.encoder(x)
        return self.latent_enc(x.view(x.shape[0], -1))

    def decode(self, x):
        x = self.latent_dec(x)
        # implement the decoding pass
        return self.decoder(x.view(-1, *self.latent_shape))

    def forward(self, x):
        # forward through the graph and return output tensor
        latent = self.encode(x)
        x = self.decode(latent)
        return x, latent
