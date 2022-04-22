# module for retrieving the data
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torch


def data(params):
    # returns list of dataloaders where each one is described in params ex:length, numbers allowed, ect
    # to be used in main.py and in training
    # please test

    # Parameters
    batch = params['batch_size']
    outliers = params['outliers']

    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=ToTensor()
    )

    indexes = []
    neg_indexes = []
    for i in range(len(mnist_testset)):
        if mnist_testset[i][1] not in outliers:
            indexes.append(i)
        else:
            neg_indexes.append(i)

    mnist_testset_pos = torch.utils.data.Subset(mnist_testset, indexes)

    test_loader = torch.utils.data.DataLoader(
        mnist_testset_pos, batch_size=batch)

    return test_loader
