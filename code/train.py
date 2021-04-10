import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision
from tqdm import tqdm
from utils import to_img 


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








        












