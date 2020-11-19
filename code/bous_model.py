class MyDataset(data.Dataset):
    def __init__(self, input, transform=None):

        self.input = input
        self.target = input
        self.transform = transform

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        ip=self.input[index]
        op=self.input[index]

        # ip=np.clip(self.input[index], 0, 1)
        # op=np.clip(self.input[index], 0, 1)

        x=self.transform(ip)
        y=self.transform(op)
        return x,y



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, 3, stride=(3,1), padding=(1,1)),  # b, 16, 150, 150
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32,4 ,stride=2, padding=1),  # b, 32, 75, 75
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # b, 64, 36, 36
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # b, 128, 18, 18
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256,4, stride=2, padding=1),  # b, 256, 9, 9
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512,4, stride=2, padding=1),  # b, 256, 4, 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        ##Decoder
        self.decoder = nn.Sequential(          
            nn.ConvTranspose2d(512, 256,5, stride=2, padding=1),  # b, 128, 9, 9
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128,4, stride=2, padding=1),  # b, 128, 18, 18
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128,64, 4, stride=2, padding=1),  # b, 64, 36, 36
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,32, 5, stride=2, padding=0),  # b, 32,75,75
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # b, 16,150,150
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16,1, 3, stride=(3,1), padding=(0,1)),  # b, 1,150,450
            nn.BatchNorm2d(1)
            # nn.Tanh()
        )

        ##Latent space
        self.h = 10
        self.down = nn.Linear(512*4*4, self.h)
        self.up = nn.Linear(self.h, 512*4*4)

    def forward(self, x):
        x = self.encoder(x)
        conv_shape = x.shape
        x = x.view(x.shape[0], -1)
        latent = self.down(x)
        x = self.up(latent)
        x = x.view(x.shape)
        x = x.view(conv_shape)
        x = self.decoder(x)
        return x