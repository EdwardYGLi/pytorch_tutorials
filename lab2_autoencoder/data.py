"""
Created by Edward Li at 2/9/21
"""
import torchvision


class MnistDataset():
    def __init__(self, storage_location):
        # create our data loaders, in a more real world example you'd have to create these dataset objects.
        # add some transforms and add some normalizations
        self.training_dataset = torchvision.datasets.MNIST(storage_location, train=True, download=True,
                                                          transform=torchvision.transforms.Compose(
                                                              # add transformations to the image here
                                                              [torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize((0.1307,), (
                                                                   0.3081,))]))
        self.test_dataset = torchvision.datasets.MNIST(storage_location, train=False, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))]))

