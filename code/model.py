import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms

class MyDataset(data.Dataset):
    def __init__(self, input, transform=None):

        self.input = input
        self.target = input
        self.transform = transform

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        ip = self.input[index]
        op = self.target[index]
        x = self.transform(ip)
        y = self.transform(op)
        return x, y

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, (3,4), stride=(1,8), padding=(1,1)),  # b, 16, 80, 80
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # b, 32, 40, 40
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # b, 64, 20, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # b, 128, 10, 10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256,4, stride=2, padding=1),  # b, 256, 5, 5
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # b, 128, 10, 10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # b, 64, 20, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # b, 32,40,40
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # b, 16,80,80
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, (3,8), stride=(1,8), padding=(1,0)),  # b, 1,80,680
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.h = 10
        self.down = nn.Linear(256*5*5, self.h)
        self.up = nn.Linear(self.h, 256*5*5)

    def forward(self, x):
        x = self.encoder(x)
        conv_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = self.down(x)
        x = self.up(x)
        x = x.view(x.shape)
        x = x.view(conv_shape)
        x = self.decoder(x)
        return x
