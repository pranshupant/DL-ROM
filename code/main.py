import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import torchvision
from model import *
from train import *
from utils import *
import warnings
import pdb
import sys

if __name__ == '__main__':

    #arguments for num_epochs and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='arg1', type=int, help="Number of Epochs")
    parser.add_argument(dest='arg2', type=int, default=16, help="Batch Size")

    args = parser.parse_args()
    num_epochs = args.arg1
    batch_size = args.arg2

    print(num_epochs, batch_size)

    #Making folders to save reconstructed images, input images and weights
    if not os.path.exists("../output"):
        os.mkdir("../output")

    if not os.path.exists("../input"):
        os.mkdir("../input")

    if not os.path.exists("../weights"):
        os.mkdir("../weights")

    warnings.filterwarnings('ignore')

    #Running the model on CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    u = np.load('../data/cylinder_u.npy', allow_pickle=True)
    # u=np.load('../data/sea_surface_noaa.npy',allow_pickle=True)
    # u = np.load('../data/cylinder_embed_200.npy', allow_pickle=True)
    print(u.shape)
    # sys.exit()
    # u = np.load('../data/boussinesq_u.npy', allow_pickle=True)
    print('Data loaded')

    #train/val split
    train_to_val = 0.85
    # rand_array = np.random.permutation(1500)
    # print(rand_array)

    u_train = u[:int(train_to_val*u.shape[0]), ...]
    u_validation = u[int(train_to_val*u.shape[0]):, ...]

    print(u_train.shape)
    print(u_validation.shape)

    # u = insert_time_channel(u, 10)
    # print(u.shape);

    img_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
    ])

    # batch_size = 16
    #Train data_loader
    train_dataset = AE_3D_Dataset(u, transform=img_transform)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)

    # print(len(train_loader))
    
    #val data_loader
    validation_dataset = AE_3D_Dataset(u, transform=img_transform)
    val_loader_args = dict(batch_size=1, shuffle=False, num_workers=4)
    val_loader = data.DataLoader(validation_dataset, **val_loader_args)

    #Loading Model
    TL = False
    if TL:
        final_model = Decode_Embedding()
        pretrained = Embedding()
        PATH = "../weights/cylinder_embed_200_t.pth"
        # PATH = "../weights/bous_500.pth"
        # pdb.set_trace()
        model = load_transfer_learning_TF(pretrained, final_model, PATH)
    else:
        model = Embedding()

    model=model.to(device)

    #Instances of optimizer, criterion, scheduler

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion=nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                    factor=0.5, patience=2, verbose=False, 
                    threshold=1e-3, threshold_mode='rel', 
                        cooldown=5, min_lr=1e-5, eps=1e-08)

    # Path='../weights/noaa_40_t.pth'

    # model.load_state_dict(torch.load(Path))
    # print(optimizer)

    #Epoch loop
    for epoch in range(num_epochs):
        start_time=time.time()
        print('Epoch no: ',epoch)
        train_loss = training_encoder(model,train_loader,criterion,optimizer)
        
        #Saving weights after every 20epochs
        if epoch%50==0: #and epoch !=0:
            output=validation_encoder(model,val_loader,criterion)
            name='../output/transformer_embed_'+str(epoch) +'.npy' 
            # #name_in='../input/'+str(epoch) +'.npy'       
            np.save(name,output)
            # del output
            # np.save(name_in,inp)

        if epoch%20==0:
            path='../weights/cylinder_embed_'+ str(epoch) +'_t.pth'
            torch.save(model.state_dict(),path)
            print(optimizer)
        
        # scheduler.step(train_loss)
        print("Time : ",time.time()-start_time)
        print('='*50)