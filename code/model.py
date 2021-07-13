import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

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

############# UNet_3D ##########################################

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
