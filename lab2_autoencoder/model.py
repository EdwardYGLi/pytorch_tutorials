import torch.nn as nn


class ConvolutionalAutoEncoder(nn.module):
    def __init__(self):
        super(ConvolutionalAutoEncoder, self).__init__()
        # define our convolutional auto-encoder here without using convolution layers

    def encode(self,x):
        # implement the encoding pass
        return x

    def decode(self,x):
        # implement the decoding pass
        return x

    def forward(self, x):
        # forward through the graph and return output tensor
        latent = self.encode(x)
        x = self.decode(latent)
        return x, latent
