

from locale import DAY_1
from turtle import forward
from h11 import Data
from matplotlib import pyplot as plt
from matplotlib.transforms import Transform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import emcee
import numpy as np


def MCMC(func):
    number_of_dimensions = 28*28
    number_of_walkers = 1
    initial_states = np.random.randn(
        number_of_walkers, number_of_dimensions)/10
    sampler = emcee.EnsembleSampler(
        nwalkers=number_of_walkers, ndim=number_of_dimensions, log_prob_fn=func)
    sampler.run_mcmc(initial_state=initial_states, nsteps=100)
    samples = sampler.get_chain(flat=True)
    return samples


transform = transforms.ToTensor()


mnist_data = datasets.MNIST(root='./data', download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(
    dataset=mnist_data, batch_size=64, shuffle=True)

dataiter = iter(data_loader)
images, labels = dataiter.next()


class Autoencodeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(
            128, 64), nn.ReLU(), nn.Linear(64, 12), nn.ReLU(), nn.Linear(12, 3))
        self.decodeur = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(
            12, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 28*28), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decodeur(encoded)
        return decoded


model = Autoencodeur()
criteron = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)

num_epochs = 10
outputs = []
for epoch in range(num_epochs):
    for (img, _) in data_loader:
        img = img.reshape(-1, 28*28)
        recon = model(img)
        loss = criteron(recon, img)
        data_MCMC = MCMC(loss)
        recon_MCMC = model(data_MCMC)
        loss_neg = criteron(data_MCMC, recon_MCMC)
        loss = loss + loss_neg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item() : 4f}')
    outputs.append((epoch, img, recon))
