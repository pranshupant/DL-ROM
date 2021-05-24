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
import pyJHTDB

from model import MyDataset, autoencoder

path_bousinessq = "../data/boussinesq.nc"
path_2dcylinder = "../data/cylinder2d.nc"
path_noaa_1990 = "../data/sst.wkmean.1981-1989.nc"
path_noaa_present="../data/sst.wkmean.1990-present.nc"


def fetchData():
    if not os.path.exists("../data"):
        os.mkdir("../data")

        cylinderURI = "https://cgl.ethz.ch/Downloads/Data/ScientificData/cylinder2d_nc.zip"
        bousinessqURI = "https://cgl.ethz.ch/Downloads/Data/ScientificData/boussinesq2d_nc.zip"

        urls = (cylinderURI, bousinessqURI)

        for url in urls:
            file_name = "../data/" + url.split('/')[-1]
            print("Fetching Data ...")
            try:
                urllib.request.urlretrieve(url,file_name)

                with ZipFile(file_name, 'r') as zipObj:
                    zipObj.extractall("../data/")

            except Exception as e:
                print(e)
            

def loadDataset(path):
    return Dataset(path)


def createAnimation(data, name):

    fig, ax = plt.subplots()
    plt.axis('off')
    ims = [[ax.imshow(data[i], animated=True)] for i in range(1, len(data))]
    
    ani = ArtistAnimation(fig, ims, interval=100 , blit=True, repeat_delay=1000)
    ani.save("%s.mp4"%name)
    writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(f"../data/{name}.mp4", writer=writer)

    # plt.show()

def createDataset(dataset, value, name):
    tdim = dataset['tdim'].shape[0]
    t = np.arange(0, tdim, 1)
    print(name)

    val = np.array(dataset[value][t,:,:])
    print(val.shape)
    print()
    
    np.save(name, val)
    plt.matshow(val[145])
    # plt.show()

def createDataset_NOAA(dataset1,dataset2,value,name):
    data1 = np.array(dataset1[value][:])
    data2 = np.array(dataset2[value][:])

    combined_data=np.concatenate([data1,data2])
    np.save(name,combined_data)

def createDataset_channel_flow():
    # 2048×512×1536
    lJHTDB = pyJHTDB.libJHTDB()
    lJHTDB.initialize()

    #Add token
    auth_token  = "edu.cmu.andrew.ppant-68a123d6"  #Replace with your own token here
    lJHTDB.add_token(auth_token)

    px = np.linspace(0, 8*np.pi, 320)
    py = np.linspace(-1, 1, 80)
    x = np.zeros((px.shape[0], py.shape[0], 3), np.float32)
    # print(t[np.newaxis, :].shape)
    x[:, :, 0] = px[:, np.newaxis]
    x[:, :, 1] = py[np.newaxis,:]
    x[:, :, 2] = 0.

    T = np.linspace(0., 25.9, 2500)
    dataset = np.zeros((len(T), 320, 80))

    for idx, time in enumerate(T):
        print(idx)
        dataset[idx] = lJHTDB.getData(time, x, data_set='channel',
            sinterp = 4, tinterp = 0,
            getFunction = 'getVelocity')[:,:,0]
    print(dataset.shape)
    np.save('../data/channel_data_2500.npy', dataset)

def loadVar(file_name):
    return np.load(file_name)

if __name__ == "__main__":

    fetchData()
    print("Ran..")

    cylinder2D = loadDataset(path_2dcylinder)
    boussinesq = loadDataset(path_bousinessq)
    # noaa_1990 = loadDataset(path_noaa_1990)
    # noaa_present = loadDataset(path_noaa_present) 
    print(cylinder2D)
    print(boussinesq)

    u_c = np.array(cylinder2D['u'])
    u_b = np.array(boussinesq['u'])

    # createAnimation(u_c, "../data/cylinder2d")
    # createAnimation(u_b, "../data/bousinessq")

    # createDataset(cylinder2D, 'u' ,'../data/cylinder_u')
    # createDataset(cylinder2D, 'v' ,'../data/cylinder_v')

    # createDataset(boussinesq, 'u' ,'../data/boussinesq_u')
    # createDataset(boussinesq, 'v' ,'../data/boussinesq_v')

    # createDataset_NOAA(noaa_1990,noaa_present, 'sst' , '../data/sea_surface_noaa')

    createDataset_channel_flow()


    u_velocityCylinder = loadVar('../data/cylinder_u.npy')
    print("Loaded variable from file: ", u_velocityCylinder.shape)

    v_velocityBousinessq = loadVar('../data/boussinesq_v.npy')
    print("Loaded variable from file: ", v_velocityBousinessq.shape)

    # sst = loadVar('../data/sea_surface_noaa.npy')
    # print("Loaded variable from file: ", sst.shape)

    channel_flow = loadVar('../data/channel_data_2500.npy')
    print(f'Created Channel flow Dataset with shape {channel_flow.shape}')


    # img_transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
    # ])

    # batch_size = 16
    # train_dataset = MyDataset(u_velocityCylinder, transform=img_transform)
    # train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4)
    # train_loader = data.DataLoader(train_dataset, **train_loader_args)
    # print(train_dataset.__len__())
    # print(train_loader.__len__())
    
    
    

