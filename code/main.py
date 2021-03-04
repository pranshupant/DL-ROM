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
from model import MyDataset, MLP_Dataset, LSTM_Dataset, autoencoder, autoencoder_B, MLP, Unet, LSTM, LSTM_B, AE_3D_Dataset, autoencoder_3D,UNet_3D
from train import training, validation
from utils import load_transfer_learning, insert_time_channel
import warnings
import pdb

if __name__ == '__main__':

    #arguments for num_epochs and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='arg1', type=int, help="Number of Epochs")
    parser.add_argument(dest='arg2', type=int, default=16, help="Batch Size")
    parser.add_argument(dest='arg3', type=str, default='2d_cylinder', help="Name of Dataset")

    args = parser.parse_args()
    num_epochs = args.arg1
    batch_size = args.arg2
    dataset_name = args.arg3

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

    if dataset_name == '2d_cylinder':
        u = np.load('../data/cylinder_u.npy', allow_pickle=True)[:-1, ...]
        v = np.load('../data/cylinder_v.npy', allow_pickle=True)[:-1, ...]

    elif dataset_name == 'boussinesq':
        u = np.load('../data/boussinesq_u.npy', allow_pickle=True)[:-1, ...]
        v = np.load('../data/boussinesq_v.npy', allow_pickle=True)[:-1, ...]

    elif dataset_name == 'SST':
        u = np.load('../data/sea_surface_noaa.npy',allow_pickle=True)[:2000, ...]
        
    else: 
        print('Dataset Not Found')
        
    
    print(f'Data Loaded in Dataset: {dataset_name} with shape {u.shape[0]}')

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
    train_dataset = AE_3D_Dataset(u_train,dataset_name,transform=img_transform)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)

    # print(len(train_loader))
    
    #val data_loader
    validation_dataset = AE_3D_Dataset(u_validation,dataset_name,transform=img_transform)
    val_loader_args = dict(batch_size=1, shuffle=False, num_workers=4)
    val_loader = data.DataLoader(validation_dataset, **val_loader_args)

    #Loading Model
    TL = False
    
    if TL:
        final_model = LSTM()
        pretrained = autoencoder()
        PATH = "../weights/1000.pth"
        # PATH = "../weights/bous_500.pth"
        # pdb.set_trace()
        model = load_transfer_learning(pretrained, final_model, PATH)
    else:
        model = UNet_3D(name=dataset_name)

    model = model.to(device)

    #Instances of optimizer, criterion, scheduler

    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                    factor=0.5, patience=2, verbose=False, 
                    threshold=1e-3, threshold_mode='rel', 
                        cooldown=5, min_lr=1e-5, eps=1e-08)

    # model.load_state_dict(torch.load(Path))
    # print(optimizer)

    #Epoch loop
    for epoch in range(num_epochs):
        start_time=time.time()
        print('Epoch no: ',epoch)
        train_loss = training(model,train_loader,criterion,optimizer)
        
        #Saving weights after every 20epochs
        if epoch%50==0 and epoch !=0:
            output=validation(model,val_loader,criterion)
            name='../output/'+dataset_name+'_'+str(epoch) +'.npy'        
            np.save(name,output)
            del output

        if epoch%20==0:
            path='../weights/'+ dataset_name + '_' + str(epoch) +'_t.pth'
            torch.save(model.state_dict(),path)
            print(optimizer)
        
        scheduler.step(train_loss)
        print("Time : ",time.time()-start_time)
        print('='*50)