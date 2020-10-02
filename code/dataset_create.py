from netCDF4 import Dataset
import numpy as np
import os
import urllib.request
import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt 
from matplotlib.animation import ArtistAnimation, FFMpegWriter
from zipfile import ZipFile

from model import MyDataset, autoencoder

path_bousinessq = "./../data/boussinesq.nc"
path_2dcylinder = "./../data/cylinder2d.nc"


def fetchData():
    if not os.path.exists("../data2"):
        os.mkdir("../data2")

        cylinderURI = "https://cgl.ethz.ch/Downloads/Data/ScientificData/cylinder2d_nc.zip"
        bousinessqURI = "https://cgl.ethz.ch/Downloads/Data/ScientificData/boussinesq2d_nc.zip"

        urls = (cylinderURI, bousinessqURI)

        for url in urls:
            file_name = "../data2/" + url.split('/')[-1]
            print("Fetching Data ...")
            try:
                urllib.request.urlretrieve(url,file_name)
                ZipFile.extractall()
            except Exception as e:
                print(e)
            

def loadDataset(path):
    return Dataset(path)


def createAnimation(data, name):

    fig, ax = plt.subplots()
    ims = [[ax.imshow(data[i], animated=True)] for i in range(1, len(data))]
    
    ani = ArtistAnimation(fig, ims, interval=100 , blit=True, repeat_delay=1000)
    # ani.save("%s.mp4"%name)
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("../data/movie.mp4", writer=writer)

    # plt.show()

def createDataset(dataset, value, name):
    tdim = dataset['tdim'].shape[0]
    t = np.arange(0, tdim, 5)
    print(name)

    val = np.array(dataset[value][t,:,:])
    print(val.shape)
    print()
    
    np.save(name, val)
    plt.matshow(val[145])
    plt.show()

def loadVar(file_name):
    return np.load(file_name)

if __name__ == "__main__":

    # fetchData()

    cylinder2D = loadDataset(path_2dcylinder)
    boussinesq = loadDataset(path_bousinessq)
    print(cylinder2D)
    print(boussinesq)

    u_c = np.array(cylinder2D['u'])
    u_b = np.array(boussinesq['u'])

    # createAnimation(u_c, "cylinder2d")
    # createAnimation(u_b, "bousinessq")

    createDataset(cylinder2D, 'u' ,'../data/cylinder_u')
    createDataset(cylinder2D, 'v' ,'../data/cylinder_v')

    createDataset(boussinesq, 'u' ,'../data/boussinesq_u')
    createDataset(boussinesq, 'v' ,'../data/boussinesq_v')

    u_velocityCylinder = loadVar('../data/cylinder_u.npy')
    print("Loaded variable from file: ", u_velocityCylinder.shape)

    v_velocityBousinessq = loadVar('../data/boussinesq_v.npy')
    print("Loaded variable from file: ", v_velocityBousinessq.shape)




    img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])

    batch_size = 16
    train_dataset = MyDataset(u_velocityCylinder, transform=img_transform)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    print(train_dataset.__len__())
    
    
    

