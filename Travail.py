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






epochs = 5
batch_size = 32
learning_rate = 1e-2

mnist_trainset = MNIST(
    root='./data', train=True, download=True, transform=ToTensor())
mnist_testset = MNIST(
    root='./data', train=False, download=True, transform=ToTensor())



indexes = []
neg_indexes = []
held_digits = [0]  # changes batch size !
for i in range(len(mnist_trainset)):
    if mnist_trainset[i][1] not in held_digits:
        indexes.append(i)
    else:
        neg_indexes.append(i)



mnist_trainset_pos = torch.utils.data.Subset(
    mnist_trainset, indexes)
mnist_trainset_neg = torch.utils.data.Subset(
    mnist_trainset, neg_indexes)


# dataloaders

# train
pos_trainloader = DataLoader(
    mnist_trainset_pos, batch_size=batch_size)
neg_trainloader = DataLoader(
    mnist_trainset_neg, batch_size=batch_size)
# test -- all digits
testloader = DataLoader(mnist_testset, batch_size=batch_size)



class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 36),
            nn.ReLU(True),
            nn.Linear(36, 18),
            nn.ReLU(True),
            nn.Linear(18, 9)
        )
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(True),
            nn.Linear(18, 36),
            nn.ReLU(True),
            nn.Linear(36, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded  


model = Autoencoder()

loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate)




def training(epochs):
    for epoch in range(epochs):
        for data in pos_trainloader:
            ''' data est une liste de deux éléments, le premier (data[0])est un regroupement d'image (liste d'image) 
            donc c'est un mini batch et le deuxième (data[1]) est la liste des entiers correspondants aux images de la liste 1'''
            images, labels = data
            
            images = images.reshape(-1, 28*28)
            assert len(images) == len(labels)

            output = model(images)
            loss = loss_function(output, images)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'epoch [{epoch + 1}/{epochs}], loss:{loss.data.item()}')


training(epochs)

def dist(im1,im2):
    n=len(im1)
    d=0
    for i in range(n):
        for j in range(n):
            d+=(im1[i][j]-im2[i][j])**2
            
    d = m.sqrt(d)
    return (d)

L = []
for (image,label) in testloader:
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



Total=len(mnist_testset)

def visu(tau):
    fake_pos = 0
    true_pos = 0
    for el in L:
        if el[0]<tau and (el[1] not in held_digits): 
            true_pos += 1
        if el[0]<tau and (el[1] in held_digits):
            fake_pos += 1
    return (fake_pos/nb_fake_pos, true_pos/nb_true_pos)


Fake_pos = []
True_pos = []

T = [moy*0.01*i for i in range(1000)]

s = 0
for tau in T:
    el = visu(tau)
    Fake_pos.append(el[0])
    True_pos.append(el[1])
    s += 1
    print(s)



plt.figure()
plt.plot(Fake_pos,True_pos)
plt.show() 





