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
from models.AE_Thomas_dMdP import Autoencoder







model = torch.load('./saved_models/AE_ThomasdMdP.pth')


