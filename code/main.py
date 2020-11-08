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
from model import MyDataset, MLP_Dataset, LSTM_Dataset, autoencoder, MLP, Unet, LSTM
from train import training,validation
import warnings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='arg1', type=int, help="Number of Epochs")
    parser.add_argument(dest='arg2', type=int, default=16, help="Batch Size")

    args = parser.parse_args()
    num_epochs = args.arg1
    batch_size = args.arg2

    print(num_epochs, batch_size)

    if not os.path.exists("../output"):
        os.mkdir("../output")

    if not os.path.exists("../input"):
        os.mkdir("../input")

    if not os.path.exists("../weights"):
        os.mkdir("../weights")

    warnings.filterwarnings('ignore')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    u_velocityCylinder = np.load('../data/cylinder_u.npy', allow_pickle=True)
    print('Data loaded')

    img_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
    ])

    # batch_size = 16
    train_dataset = LSTM_Dataset(u_velocityCylinder, transform=img_transform)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    
    
    validation_dataset=LSTM_Dataset(u_velocityCylinder, transform=img_transform)
    val_loader_args = dict(batch_size=1, shuffle=False, num_workers=4)
    val_loader = data.DataLoader(validation_dataset, **val_loader_args)


    model= LSTM()
    model=model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion=nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                    factor=0.1, patience=5, verbose=False, 
                    threshold=1e-3, threshold_mode='rel', 
                        cooldown=5, min_lr=1e-4, eps=1e-08)

    for epoch in range(num_epochs):
        start_time=time.time()
        print('Epoch no: ',epoch)
        train_loss = training(model,train_loader,criterion,optimizer)
        # scheduler.step(epoch)
        if epoch%20==0:
            inp, output=validation(model,val_loader,criterion)
            name='../output/'+str(epoch) +'.npy' 
            name_in='../input/'+str(epoch) +'.npy'       
            np.save(name,output)
            np.save(name_in,inp)
            path='../weights/'+ str(epoch) +'.pth'
            torch.save(model.state_dict(),path)
            print(optimizer)
        
        scheduler.step(train_loss)
        print("Time : ",time.time()-start_time)
        print('='*50)