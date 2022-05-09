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
import time
import time
import sys


def visu(params, dataloader, models):
    #   params are the type of results
    #   plot ?
    #   def get_detector ?

    held_digits = params['outliers']
    visu_choice = params['visu_choice']

    if visu_choice == "roc":
        affichage_roc(held_digits, dataloader, models[0], "roc")

    elif visu_choice == "tab":
        Y = []
        for i in range(len(held_digits)):
            Y.append(affichage_roc(held_digits[i], dataloader, models[i], "tab"))
        tab = [held_digits, Y]
        form = "{0:10}{1:10}"
        for val in tab:
            print(form.format(*val))

    else:
        print("visu_choice has to be either roc or tab")


def affichage_roc(held_digits, dataloader, model, choice):
    L = []
    for (image, label) in dataloader:
        for i in range(len(image)):
            im1 = image[i][0].reshape(-1, 28*28)
            im2 = model(im1)
            d = dist(im1, im2)
            L.append((d, label[i]))
    moy = np.mean([L[i][0]for i in range(len(L))])
    print("distances", L)
    nb_fake_pos = 0
    nb_true_pos = 0
    x1, x2 = 0, 0
    y1, y2 = 0, 0
    aire = 0

    for el in L:
        if int(el[1]) in held_digits:
            nb_fake_pos += 1
        else:
            nb_true_pos += 1

    Total = len(L)

    Fake_pos = []
    True_pos = []

    T = [moy*0.01*i for i in range(1000)]

    s = 0
    for tau in T:
        el = visualize(tau, held_digits, nb_fake_pos, nb_true_pos, L)
        Fake_pos.append(el[0])
        True_pos.append(el[1])
        x2 = nb_fake_pos
        y2 = nb_true_pos
        aire += (x2 - x1) * (y2 + y1) / 2
        x1, y1 = x2, y2
        s += 1
        sys.stdout.write('\rloading |  {}/{}'.format(s, len(T)))
    sys.stdout.write('\rDone!     ')

    if choice == "roc":
        plt.figure()
        plt.plot(Fake_pos, True_pos, label='evaluation')
        plt.plot([0, 1], [0, 1], label="no skill")
        plt.xlabel("False positive rate")
        plt.ylabel('True positive rate')
        plt.title('Roc curve - Efficiency of the autoencoder')
        plt.legend()
        plt.show()

    if choice == "tab":
        return aire


def dist(im1, im2):
    n = len(im1)
    d = 0
    for i in range(n):
        for j in range(n):
            d += (im1[i][j]-im2[i][j])**2

    d = m.sqrt(d)
    return (d)


def visualize(tau, held_digits, nb_fake_pos, nb_true_pos, L):
    fake_pos = 0
    true_pos = 0
    for el in L:
        if el[0] < tau and (el[1] not in held_digits):
            true_pos += 1
        if el[0] < tau and (el[1] in held_digits):
            fake_pos += 1
    return (fake_pos/nb_fake_pos, true_pos/nb_true_pos)
