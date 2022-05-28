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
from tabulate import tabulate


def visu(params, dataloader, models):
    #   params are the type of results
    #   plot ?
    #   def get_detector ?

    held_digits = params['outliers']
    visu_choice = params['visu_choice']

    if visu_choice == "roc":
        if len(held_digits) > 1:
            for i in range(len(held_digits)):
                affichage_roc([held_digits[i]], dataloader,
                              models[i], "roc", "outliers")

        elif len(models) > 1:

            for i in range(len(models)):

                affichage_roc([held_digits[0]], dataloader,
                              models[i], "roc", "models", i+1)
        else:
            affichage_roc([held_digits[0]], dataloader, models[0], "roc")

        plt.plot([0, 1], [0, 1], label="no skill")
        plt.xlabel("False positive rate")
        plt.ylabel('True positive rate')
        plt.title('Roc curve - Efficiency of the autoencoder')
        plt.legend()
        plt.show()

    elif visu_choice == "tab":
        Y = []
        # print(models)
        # print(held_digits)
        # print(len(held_digits))
        for i in range(len(held_digits)):
            Y.append(
                round(affichage_roc([held_digits[i]], dataloader, models[i], "tab"), 3))
        tab = [['Area'] + held_digits, ['Performance'] + Y]
        print(tabulate(tab))

    else:
        print("visu_choice has to be either roc or tab")


def affichage_roc(held_digits, dataloader, model, choice, criterion="outliers", number=0):
    L = []
    for (image, label) in dataloader:
        for i in range(len(image)):
            im1 = image[i][0].reshape(-1, 28*28)
            im2 = model(im1)
            d = dist(im1, im2)
            L.append((d, label[i]))
    moy = np.mean([L[i][0]for i in range(len(L))])
    nb_fake_pos = 0
    nb_true_pos = 0
    x1, x2 = 0, 0
    y1, y2 = 0, 0
    aire = 0
    abs = 0
    ord = 0

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
    compt = 0

    for tau in T:
        compt += 1
        el = visualize(tau, held_digits, nb_fake_pos, nb_true_pos, L)
        Fake_pos.append(el[0])
        True_pos.append(el[1])
        x2 = el[0]
        y2 = el[1]
        abs += x2 - x1
        ord += y2 - y1
        aire += (x2 - x1) * (y2 + y1) / 2
        x1, y1 = x2, y2
        s += 1
        if compt % 50 == 0:
            print(compt, '/ 1000')

    if choice == "roc":

        if criterion == "outliers":

            plt.plot(Fake_pos, True_pos, label='evaluation with ' +
                     str(held_digits[0]) + ' as an outlier')

        if criterion == "models":

            plt.plot(Fake_pos, True_pos, label='evaluation with the ' +
                     ordinal(number) + ' model')

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


# just gives the ordinal version of a number (e.g 1st for 1)
def ordinal(n): return "%d%s" % (
    n, "tsnrhtdd"[(n//10 % 10 != 1)*(n % 10 < 4)*n % 10::4])
