import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


# Tranformation images to pytorch tensor
tensor_transform = transforms.ToTensor()

dataset = MNIST('./data', train=True, download=True,
                transform=tensor_transform)


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
        return decoded  # , latent


epochs = 5
batch_size = 32
learning_rate = 1e-2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Autoencoder()
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate)


def training(epochs):
    for epoch in range(epochs):
        for data in dataloader:
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


def affichage_images():

    for (image, _) in dataloader:

        for i in range(len(image)):
            fig = plt.figure()
            image_mod = image[i][0].reshape(-1, 28*28)
            reconstructed_mod = model(image_mod)
            reconstructed_mod = reconstructed_mod.reshape(28, 28)
            reconstructed_mod = reconstructed_mod.detach().numpy()
            fig.add_subplot(1, 2, 1)
            plt.imshow(image[i][0])
            plt.title("Image initiale")
            fig.add_subplot(1, 2, 2)
            plt.imshow(reconstructed_mod)
            plt.title("Image reconstruite")
            plt.show()


# affichage_images()

def enregistrement():
    torch.save(model, "AE_tim.pth")


enregistrement()
