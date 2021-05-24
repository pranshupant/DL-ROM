import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision
from tqdm import tqdm
from utils import to_img, MSE_simulate


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def training(model,train_loader,criterion,optimizer):
    '''
    Training one epoch of the model 

    return: Training loss of one epoch
    '''
    model.train()
    avg_loss=[]

    for batch_num, (feats, labels) in tqdm(enumerate(train_loader), total=len(train_loader), ascii=True):
        feats, labels = feats.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(feats)
        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())
            
        torch.cuda.empty_cache()
        del feats
        del labels
        del loss

    print('Train Loss: {:.6f}'.format(sum(avg_loss)/len(avg_loss)))
    return sum(avg_loss)/len(avg_loss)


def validation(model,test_loader,criterion):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    avg_loss=[]
    
    for batch_num, (feats, labels) in tqdm(enumerate(test_loader), total=len(test_loader), ascii=True):
        feats, labels = feats.to(device), labels.to(device)

        outputs=model(feats)
        loss=criterion(outputs,labels)
        avg_loss.append(loss.item())

        del feats
        del labels
    model.train()
    print('Validation Loss: {:.4f}'.format(sum(avg_loss)/len(avg_loss)))

    return (sum(avg_loss)/len(avg_loss))

def test(model, test_loader):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    out = []
    label = []
    
    for batch_num, (feats, labels) in tqdm(enumerate(test_loader), total=len(test_loader), ascii=True):
        feats = feats.to(device)

        outputs=model(feats)
        out.append(outputs[0, 0, -1].detach().cpu().numpy()) ## Moudularize
        label.append(labels[0, 0, -1].numpy())
        # print(labels[0, 0, -1].numpy().shape)
        # print(outputs[0, 0, -1].detach().cpu().numpy().shape)

        del feats
        del labels

    return np.array(label), np.array(out)

def simulate(model, u_valid, transform):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    out = []
    label = []
    mse = []

    context = [i for i in u_valid[0:100:10]]

    for epoch in range((u_valid.shape[0]-100)//10):
        print(f"Epoch: {epoch}")
        inp = np.array(context)

        # inp=torch.from_numpy(inp)
        inp = transform(inp)
        inp=inp.permute(1, 2, 0)
        inp=inp.unsqueeze(0).unsqueeze(0)
        # print(inp.shape)

        feats = inp.to(device)

        outputs=model(feats)
        op = outputs[0, 0, -1].detach().cpu().numpy()
        context.append(op) ## Moudularize
        out.append(op)
        label.append(u_valid[100 + (10*epoch)])
        mse.append(MSE_simulate(op, u_valid[100 + (10*epoch)]))
        context.pop(0)

        del feats

    return np.array(label), np.array(out), np.array(mse)






        












