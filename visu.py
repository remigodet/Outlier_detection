# tools for testing trained models perfs and creating an optimal outlier detector

# function to be determined ...
from cmath import sqrt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import math as m


def visu(params, dataloader, model):
#   params are the type of results
#   plot ?
# def get_detector ?

    held_digits = params['outliers']

    L = []
    for (image,label) in dataloader:
        for i in range (len(image)):
            im1 = image[i][0].reshape(-1,28*28)
            im2 = model(im1)
            d = dist(im1,im2)
            L.append((d,label[i]))

    moy = np.mean([L[i][0]for i in range (len(L))])

    nb_fake_pos = 0
    nb_true_pos = 0
    for el in L:
        if el[1] in held_digits:
            nb_fake_pos += 1
        else:
            nb_true_pos += 1



    Total=len(L)

    Fake_pos = []
    True_pos = []

    T = [moy*0.01*i for i in range(1000)]

    s = 0
    for tau in T:
        el = visu(tau, held_digits, nb_fake_pos, nb_true_pos, L)
        Fake_pos.append(el[0])
        True_pos.append(el[1])
        s += 1
        print(s)



    plt.figure()
    plt.plot(Fake_pos,True_pos)
    plt.show() 

    return True


def dist(im1,im2):
    n=len(im1)
    d=0
    for i in range(n):
        for j in range(n):
            d+=(im1[i][j]-im2[i][j])**2
            
    d = m.sqrt(d)
    return (d)

def visu(tau, held_digits, nb_fake_pos, nb_true_pos, L):
    fake_pos = 0
    true_pos = 0
    for el in L:
        if el[0]<tau and (el[1] not in held_digits): 
            true_pos += 1
        if el[0]<tau and (el[1] in held_digits):
            fake_pos += 1
    return (fake_pos/nb_fake_pos, true_pos/nb_true_pos)



