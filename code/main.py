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
from train import training, validation, test
from utils import load_transfer_learning, insert_time_channel, find_weight, save_loss
import warnings
import pdb

'''
python main.py 100 32 -d_set 2d_cylinder --train/ --test -test_epoch 
'''

if __name__ == '__main__':

    #arguments for num_epochs and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='num_epochs', type=int, help="Number of Epochs")
    parser.add_argument(dest='batch_size', type=int, default=16, help="Batch Size")
    parser.add_argument('-d_set', dest='dataset', type=str, default='2d_cylinder', help="Name of Dataset")
    parser.add_argument('-test_epoch', dest='test_epoch', type=int, default=None, help="Epoch for testing")
    parser.add_argument('--test', dest='testing', action='store_true')
    parser.add_argument('--train', dest='training', action='store_true')
    parser.add_argument('--transfer', dest='transfer', action='store_true')

    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dataset_name = args.dataset
    test_epoch = args.test_epoch
    transfer_learning = args.transfer

    print(num_epochs, batch_size)

    if not os.path.exists(f'../results/{dataset_name}'):
        os.mkdir(f'../results/{dataset_name}')

    #Making folders to save reconstructed images, input images and weights
    if not os.path.exists(f'../results/{dataset_name}/output/'):
        os.mkdir(f'../results/{dataset_name}/output/')


    if not os.path.exists(f'../results/{dataset_name}/weights/'):
        os.mkdir(f'../results/{dataset_name}/weights/')

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

    elif dataset_name == '2d_cylinder_CFD':
        u_comp = np.load('../data/Vort100.npz', allow_pickle=True)
        # u_comp = np.load('../data/Velocity160.npz', allow_pickle=True)
        
        u_flat = u_comp['arr_0']
        u = u_flat.reshape(u_flat.shape[0], 320, 80)
        u = np.transpose(u, (0, 2, 1)).astype(np.float32)
        
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

    
    if transfer_learning:
        print('Using Transfer Learning')
        final_model = LSTM()
        pretrained = autoencoder()
        PATH = "../weights/1000.pth"
        # PATH = "../weights/bous_500.pth"
        # pdb.set_trace()
        model = load_transfer_learning(pretrained, final_model, PATH)
    else:
        model = UNet_3D(name=dataset_name)

    model = model.to(device)

    if args.training:
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

        #Instances of optimizer, criterion, scheduler

        optimizer = optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                        factor=0.5, patience=2, verbose=False, 
                        threshold=1e-3, threshold_mode='rel', 
                            cooldown=5, min_lr=1e-5, eps=1e-08)

        # model.load_state_dict(torch.load(Path))
        # print(optimizer)

        val_loss = {}
        #Epoch loop
        for epoch in range(num_epochs):
            start_time=time.time()
            print('Epoch no: ',epoch)
            train_loss = training(model,train_loader,criterion,optimizer)
            
            #Saving weights after every 20epochs
            if epoch%10==0:# and epoch !=0:
                val_loss[epoch] = validation(model,val_loader,criterion)

            if epoch%10==0:# and epoch != 0:
                path=f'../results/{dataset_name}/weights/{epoch}.pth'
                torch.save(model.state_dict(),path)
                print(optimizer)
            
            scheduler.step(train_loss)
            print("Time : ",time.time()-start_time)
            print('='*100)
            print()

        save_loss(val_loss, dataset_name)


    if args.testing:

        PATH = find_weight(dataset_name, test_epoch)

        print(PATH)

        model.load_state_dict(torch.load(PATH))

        test_dataset = AE_3D_Dataset(u_validation,dataset_name,transform=img_transform)
        test_loader_args = dict(batch_size=1, shuffle=False, num_workers=4)
        test_loader = data.DataLoader(test_dataset, **test_loader_args)

        labels, preds = test(model, test_loader)
        name=f'../results/{dataset_name}/output/labels.npy'        
        np.save(name, labels)

        name=f'../results/{dataset_name}/output/predictions.npy'
        np.save(name, preds)