# module for retrieving the data
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torch


def data(params):
    # returns list of dataloaders where each one is described in params ex:length, numbers allowed, ect
    # to be used in main.py and in training
    # please test

    # Parameters default value
    batch_size = 50
    outliers = []
    dataset = 'test'
    use_negative_index = False

    if 'batch_size' in params:
        batch_size = params['batch_size']
    if 'outliers' in params:
        outliers = params['outliers']
    if 'dataset' in params:
        dataset = params['dataset']
    if 'use_negative' in params:
        # If True renvoie le tuple des images à indexs positifs et négatifs
        use_negative_index = params['use_negative_index']

    if dataset == 'test':

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
        test_loader_pos = torch.utils.data.DataLoader(
            mnist_testset_pos, batch_size=batch_size)

        mnist_testset_neg = torch.utils.data.Subset(mnist_testset, neg_indexes)
        test_loader_neg = torch.utils.data.DataLoader(
            mnist_testset_neg, batch_size=batch_size)

        if use_negative_index:
            return test_loader_pos, test_loader_neg
        return test_loader_pos

    if dataset == 'train':
        mnist_trainset = datasets.MNIST(
            root='./data', train=True, download=True, transform=ToTensor())

        indexes = []
        neg_indexes = []
        for i in range(len(mnist_trainset)):
            if mnist_trainset[i][1] not in outliers:
                indexes.append(i)
            else:
                neg_indexes.append(i)

        mnist_trainset_pos = torch.utils.data.Subset(mnist_trainset, indexes)
        train_loader_pos = torch.utils.data.DataLoader(
            mnist_trainset_pos, batch_size=batch_size)

        mnist_trainset_neg = torch.utils.data.Subset(
            mnist_trainset, neg_indexes)
        train_loader_neg = torch.utils.data.DataLoader(
            mnist_trainset_neg, batch_size=batch_size)

        if use_negative_index:
            return train_loader_pos, train_loader_neg
        return train_loader_pos
