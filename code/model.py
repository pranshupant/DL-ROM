import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

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
            nn.LeakyReLU()
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
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, (3,8), stride=(1,8), padding=(1,0)),  # b, 1,80,680
            # nn.BatchNorm2d(1),
            # nn.Tanh()
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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        
        self.encoder=nn.Sequential(
            nn.Linear(640*80,4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.h=10
        self.down=nn.Linear(256,self.h)
        self.up= nn.Linear(self.h,256)
        self.decoder=nn.Sequential(
            nn.Linear(256,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096,80*640),
            nn.BatchNorm1d(80*640),
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        x=self.encoder(x)
        x=self.down(x)
        x=self.up(x)
        x=self.decoder(x)

        return x

#######################################

class Downsample(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=4,stride=2,padding=1):
        super(Downsample, self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x=self.net(x)
        return x


class Upsample(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=4,stride=2,padding=1):
        super(Upsample, self).__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channel)
        )
        self.lRelu = nn.LeakyReLU()

    def forward(self,x1,x2,last=False):
        x=torch.cat((x1,x2),dim=1)
        x=self.net(x)
        if last:
            x=x
        else:
            x=self.lRelu(x)
        return x



class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        #encoder
        self.d1=Downsample(1,16,(3,4),(1,8),(1,1))
        self.d2=Downsample(16,32)
        self.d3=Downsample(32,64)
        self.d4=Downsample(64,128)
        self.d5=Downsample(128,256)

        self.h = 10
        self.down = nn.Linear(256*5*5, self.h)
        self.up = nn.Linear(self.h, 256*5*5)

        self.u1=Upsample(512,128)
        self.u2=Upsample(256,64)
        self.u3=Upsample(128,32)
        self.u4=Upsample(64,16)
        self.u5=Upsample(32,1,(3,8),(1,8),(1,0))

    def forward(self,x):

        down1=self.d1(x)
        down2=self.d2(down1)
        down3=self.d3(down2)
        down4=self.d4(down3)
        down5=self.d5(down4)

        conv_shape = down5.shape
        mid = down5.view(down5.shape[0], -1)
        mid = self.down(mid)
        mid= F.relu(mid)
        mid = self.up(mid)
        mid= F.relu(mid)
        mid = mid.view(mid.shape)
        mid = mid.view(conv_shape)

        up1=self.u1(down5,mid)
        up2=self.u2(down4,up1)
        up3=self.u3(down3,up2)
        up4=self.u4(down2,up3)
        up5=self.u5(down1,up4,last=True)

        return up5


