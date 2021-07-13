import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

#Dataset class for MLP model
class MLP_Dataset(data.Dataset):
    def __init__(self, input, transform=None):

        self.input = input
        self.target = input
        self.transform = transform

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        ip=self.input[index].flatten()
        op=self.input[index].flatten()

        # ip=np.clip(self.input[index], 0, 1)
        # op=np.clip(self.input[index], 0, 1)

        x=torch.from_numpy(ip).float()
        y=torch.from_numpy(op).float()
        return x,y

#Dataset class for CNN Autoencoders
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

#Dataset class for CNN LSTM model
class LSTM_Dataset(data.Dataset):
    def __init__(self, input, transform=None):

        self.input = input#[:-5]
        self.target = input#[5:]
        self.transform = transform

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        ip=self.input[index]
        op=self.target[index]

        # ip=np.clip(self.input[index], 0, 1)
        # op=np.clip(self.input[index], 0, 1)

        x=self.transform(ip)
        y=self.transform(op)
        return x,y

class AE_3D_Dataset(data.Dataset):
    def __init__(self, input, name='2d_cylinder', transform=None):
        # if name == 'SST':
        #     input = input[:,10:-10,20:-20]

        self.input = input[:-10]
        self.target = input[10:]
        self.transform = transform
        self.hashmap = {i:range(i, i+100, 10) for i in range(self.input.shape[0] - 100)}
        print(len(self.hashmap))

    def __len__(self):
        return len(self.hashmap)

    def __getitem__(self, index):
        idx = self.hashmap[index]
        # print(idx)
        idy = self.hashmap[index]
        ip=self.input[idx]
        op=self.target[idy]

        x=self.transform(ip)
        x=x.permute(1, 2, 0)
        x=x.unsqueeze(0)

        y=self.transform(op)
        y=y.permute(1, 2, 0)
        y=y.unsqueeze(0)
        return x,y


######## Skip-Blocks FOR ResNet TYPE ARCHITECTURE ################################################
class BasicBlock_Up(nn.Module):
  def __init__(self, in_channel,stride=1):
      super(BasicBlock_Up, self).__init__()
      self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(in_channel)
      self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn2 = nn.BatchNorm2d(in_channel)

  def forward(self, x):
      identity = x
      out = F.leaky_relu(self.bn1(self.conv1(x)))
      out = F.leaky_relu(self.bn2(self.conv2(out)))
      out += identity
      out = F.leaky_relu(out,inplace=True)
      return out

class BasicBlock_down(nn.Module):
  def __init__(self, in_channel,stride=1):
      super(BasicBlock_down, self).__init__()
      self.conv1 = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=3, stride=stride,padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(in_channel)
      self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn2 = nn.BatchNorm2d(in_channel)

  def forward(self, x):
      identity = x
      out = F.leaky_relu(self.bn1(self.conv1(x)))
      out = F.leaky_relu(self.bn2(self.conv2(out)))
      out += identity
      out = F.leaky_relu(out,inplace=True)
      return out

###################################################################

######### CNN Autoencoder model ##################################
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, (3,4), stride=(1,8), padding=(1,1)),  # b, 16, 80, 320
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32,4 ,stride=2, padding=1),  # b, 32, 40, 40
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

        ##Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128,4, stride=2, padding=1),  # b, 128, 10, 10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128,64, 4, stride=2, padding=1),  # b, 64, 20, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,32, 4, stride=2, padding=1),  # b, 32,40,40
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # b, 16,80,80
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16,1, (3,8), stride=(1,8), padding=(1,0)),  # b, 1,80,680
            nn.BatchNorm2d(1)
            # nn.Tanh()
        )

        ##Latent space
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

################################################################

######### Simple Autoencoder ###################################
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

        ##Latent Space
        self.h=10
        self.down=nn.Linear(256,self.h)
        self.up= nn.Linear(self.h,256)

        ##Decoder
        self.decoder=nn.Sequential(
            nn.Linear(256,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096,80*640),
            nn.BatchNorm1d(80*640),
        )
    
    def forward(self,x):
        x=self.encoder(x)
        x=self.down(x)
        x=self.up(x)
        x=self.decoder(x)

        return x

#################################################

########## U-NET model#############################

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

######### CNN Autoencoder model - Bousinessq ##################################
class autoencoder_B(nn.Module):
    def __init__(self):
        super(autoencoder_B, self).__init__()
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



############ CNN-LSTM Autoencoder ######################
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM ,self).__init__()
        #LSTM layers
        self.lstm1 = nn.GRU(10, 32, 2, bidirectional=False, batch_first=True)
        self.lstm2 = nn.GRU(32, 10, 2, bidirectional=False, batch_first=True)
        # self.lstm3 = nn.GRU(64, 10, 2, bidirectional=False, batch_first=True)
        
        ##Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, (3,4), stride=(1,8), padding=(1,1)),  # b, 16, 80, 320
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32,4 ,stride=2, padding=1),  # b, 32, 40, 40
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

        ##Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128,4, stride=2, padding=1),  # b, 128, 10, 10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128,64, 4, stride=2, padding=1),  # b, 64, 20, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,32, 4, stride=2, padding=1),  # b, 32,40,40
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # b, 16,80,80
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16,1, (3,8), stride=(1,8), padding=(1,0)),  # b, 1,80,680
            nn.BatchNorm2d(1)
            # nn.Tanh()
        )

        ##Latent space
        self.h = 10

        self.down = nn.Linear(256*5*5, self.h)
        self.up = nn.Linear(self.h, 256*5*5)

    def forward(self, x):
        x = self.encoder(x)
        conv_shape = x.shape
        x = x.view(x.shape[0], -1)
        # print(x.size())
        x = self.down(x)
        # print(x.size())
        x = x.view(1,x.shape[0],self.h)
        # print(x.size())
        x = self.lstm1(x)[0]
        x = self.lstm2(x)[0]
        # x = self.lstm3(x)[0]
        # print(x.shape)
        x = x.view(conv_shape[0],self.h)
        x = self.up(x)
        x = x.view(conv_shape)
        x = self.decoder(x)
        return x
########################################
class LSTM_model(nn.Module):
    def __init__(self):
        super(LSTM_model ,self).__init__()
        #LSTM layers
        self.lstm1 = nn.LSTM(10, 64, 3, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(64, 10, 3, bidirectional=False, batch_first=True)

    def forward(self, x):
        x = self.lstm1(x)[0]
        x = self.lstm2(x)[0]
        x=x[-1,:]
        x=x[-1,:]
        return x

############ CNN-LSTM_B Autoencoder ######################
class LSTM_B(nn.Module):
    def __init__(self):
        super(LSTM_B ,self).__init__()
        #LSTM layers
        self.lstm1 = nn.LSTM(10, 64, 3, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(64, 10, 3, bidirectional=False, batch_first=True)
        
        ##Encoder
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

    # def forward(self, x):
    #     x = self.encoder(x)
    #     conv_shape = x.shape
    #     x = x.view(x.shape[0], -1)
    #     latent = self.down(x)
    #     x = self.up(latent)
    #     x = x.view(x.shape)
    #     x = x.view(conv_shape)
    #     x = self.decoder(x)
    #     return x

    def forward(self, x):
        x = self.encoder(x)
        conv_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = self.down(x)
        x = x.view(x.shape[0],1,self.h)
        x = self.lstm1(x)[0]
        x = self.lstm2(x)[0]
        # print(x.shape)
        x = x.view(x.shape[0],self.h)
        x = self.up(x)
        x = x.view(conv_shape)
        x = self.decoder(x)
        return x
########################################

######### CNN Autoencoder model ##################################
class autoencoder_3D(nn.Module):
    def __init__(self):
        super(autoencoder_3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, (3, 3, 4), stride=(1, 1, 8), padding=(0, 1, 1)),  # b, 8, 16, 80, 320
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 32, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)),  # b, 6, 32, 40, 40
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)),  # b, 4, 64, 20, 20
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)),  # b, 2, 128, 10, 10
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 256, (2, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)),  # b, 1, 256, 5, 5
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
        )

        ##Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128,(2, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)),  # b, 2, 128, 10, 10
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128,64, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)),  # b, 4, 64, 20, 20
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64,32, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)),  # b, 6, 32,40,40
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32, 16, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)),  # b, 8, 16, 80, 80
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(16, 1, (3, 3, 8), stride=(1, 1, 8), padding=(0, 1, 0)),  # b, 10, 1, 80, 680
            nn.BatchNorm3d(1)
            # nn.Tanh()
        )

        ##Latent space
        self.h = 32
        self.down = nn.Linear(256*5*5, self.h)
        self.up = nn.Linear(self.h, 256*5*5)

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        conv_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = self.down(x)
        x = self.up(x)
        # print(x.shape)
        # x = x.view(x.shape)
        x = x.view(conv_shape)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        return x

################################################################

############# UNet_3D #########################################

class Downsample_3d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel,stride,padding=(0,1,1)):
        super(Downsample_3d, self).__init__()
        self.net=nn.Sequential(
            nn.Conv3d(in_channel,in_channel,kernel_size=kernel,stride=stride,padding=padding,groups=in_channel),
            nn.Conv3d(in_channel,out_channel,1,1,0),
            nn.BatchNorm3d(out_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x=self.net(x)
        return x


class Upsample_3d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel,stride,padding=(0,1,1)):
        super(Upsample_3d, self).__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose3d(in_channel,in_channel,kernel_size=kernel,stride=stride,padding=padding,groups=in_channel),
            nn.ConvTranspose3d(in_channel,out_channel,1,1,0),
            nn.BatchNorm3d(out_channel)
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

class UNet_3D(nn.Module):
    def __init__(self,name):
        super(UNet_3D, self).__init__()
        self.name=name

        if name=='2d_cylinder_CFD' or name=='2d_cylinder' or name=='2d_sq_cyl':
            d1=Downsample_3d(1, 16, (3, 3, 8), stride=(1, 1, 4), padding=(0, 1, 2)) #16,80,80
            u5=Upsample_3d(32, 1, (3, 3, 8), stride=(1, 1, 4), padding=(0, 1, 2)) #190,360

        elif name=='2d_airfoil':
            d1=Downsample_3d(1, 16, (3, 3, 4), stride=(1, 1, 2), padding=(0, 1, 1)) #16,80,80
            u5=Upsample_3d(32, 1, (3, 3, 4), stride=(1, 1, 2), padding=(0, 1, 1)) #190,360
        
        elif name=='boussinesq':
            d1= Downsample_3d(1,16,(3,8,4),stride=(1,4,2),padding=(0,2,1))
            u5 = Upsample_3d(32,1,(3,8,4),stride=(1,4,2),padding=(0,2,1))
            # u6 = nn.ConvTranspose3d(8,1,(3,6,3),stride=(1,3,1),padding=(1,0,1))
            # self.u6=u6

        elif name=='SST' or name=='2d_plate':
            #Note - Remember to crop in dataloader
            d1=Downsample_3d(1, 16, (3, 4, 8), stride=(1, 2, 4), padding=(0, 1, 2))
            u5=Upsample_3d(32,1,(3,4,8),stride=(1,2,4),padding=(0,1,2))
            
        elif name=='channel_flow':
            d1=Downsample_3d(1, 16, (3, 8, 3), stride=(1, 4, 1), padding=(0, 2, 1)) #16,80,80
            u5=Upsample_3d(32, 1, (3, 8, 3), stride=(1, 4, 1), padding=(0, 2, 1)) #190,360

        else:
            print(f'Dataset Not Defined')


        self.d1=d1
        self.d2=Downsample_3d(16, 32, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #44
        self.d3=Downsample_3d(32, 64, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #22 
        self.d4=Downsample_3d(64, 128, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #10
        self.d5=Downsample_3d(128, 256, (2, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #5

        self.h = 32
        self.down = nn.Linear(256*5*5, self.h)
        self.up = nn.Linear(self.h, 256*5*5)

        self.u1=Upsample_3d(512, 128,(2, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #10
        self.u2=Upsample_3d(256,64, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #22
        self.u3=Upsample_3d(128,32, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #44
        self.u4=Upsample_3d(64, 16, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #90
        self.u5=u5#190,360


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
        
        # if self.name=='boussinesq':
        #     out= self.u5(down1,up4)
        #     out =self.u6(out)

        # else:
        out = self.u5(down1,up4,last=True)

        return out
