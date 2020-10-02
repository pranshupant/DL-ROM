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
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # b, 32, 32, 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # b, 128, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # b, 64, 32, 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # b, 1, 64, 64
            nn.BatchNorm2d(1),
            nn.Tanh()
        )
        self.h = 6
        self.down = nn.Linear(128*8*8, self.h)
        self.up = nn.Linear(self.h, 128*8*8)

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
