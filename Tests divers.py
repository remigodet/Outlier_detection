from unittest import TestLoader
import visu as visu
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from models.AE_ThomasdMdP import Autoencoder


params = {'outliers': [0], 'visu_choice': 'roc'}

model = torch.load('./saved_models/AE_ThomasdMdP-NA.pth')

mnist_testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=50)


visu.visu(params, dataloader, [model])
