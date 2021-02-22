import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision
from utils import to_img 


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def training_encoder(model,train_loader,criterion,optimizer):
    '''
    Training one epoch of the model 

    return: Training loss of one epoch
    '''
    model.train()
    avg_loss=[]

    for batch_num, (src, labels) in enumerate(train_loader):
        src, labels = src.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs= model(src)
        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())

        torch.cuda.empty_cache()
        del src
        del labels
        del loss

    print('Train Loss: {:.6f}'.format(sum(avg_loss)/len(avg_loss)))
    return sum(avg_loss)/len(avg_loss)

def training_tf(model,train_loader,criterion,optimizer):
    '''
    Training one epoch of the model 

    return: Training loss of one epoch
    '''
    model.train()
    avg_loss=[]

    for batch_num, (src, tgt, labels) in enumerate(train_loader):
        src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)
        
        optimizer.zero_grad()
        src=src.permute(1,0,2)
        tgt=tgt.permute(1,0,2)
        outputs = model(src, tgt)
        loss=criterion(outputs.permute(1,0,2),labels)

        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())

        torch.cuda.empty_cache()
        del src
        del tgt
        del labels
        del loss

    print('Train Loss: {:.6f}'.format(sum(avg_loss)/len(avg_loss)))
    return sum(avg_loss)/len(avg_loss)



def validation_encoder(model,test_loader,criterion):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    avg_loss=[]
    out=[]
    inp=[]
    
    for batch_num, (src, labels) in enumerate(test_loader):
        src, labels = src.to(device), labels.to(device)


        outputs= model(src)
        temp=outputs[0].detach().cpu().numpy()
        out.append(temp)
        # temp2=feats[0].detach().cpu().numpy()
        # inp.append(temp2.reshape(-1, 180,360))
        loss=criterion(outputs,labels)
        avg_loss.append(loss.item())

        # outputs=torch.squeeze(outputs,1)
        # out.append(outputs.detach().cpu().numpy())

        del src
        # del tgt
        # del temp
        # del temp2
        del labels
    model.train()
    print('Validation Loss: {:.4f}'.format(sum(avg_loss)/len(avg_loss)))

    return np.array(out)


def validation_tf(model,test_loader,criterion):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    avg_loss=[]
    out=[]
    inp=[]
    
    for batch_num, (src, tgt, labels) in enumerate(test_loader):
        src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)

        src=src.permute(1,0,2)
        tgt=tgt.permute(1,0,2)

        outputs = model(src, tgt)
        outputs=outputs.permute(1,0,2)
        # temp=hidden[0].detach().cpu().numpy()
        # out.append(temp)
        # temp2=feats[0].detach().cpu().numpy()
        # inp.append(temp2.reshape(-1, 180,360))
        loss=criterion(outputs,labels)
        avg_loss.append(loss.item())

        outputs=torch.squeeze(outputs,1)
        out.append(outputs.detach().cpu().numpy())

        del src
        del tgt
        # del temp
        # del temp2
        del labels
    model.train()
    print('Validation Loss: {:.4f}'.format(sum(avg_loss)/len(avg_loss)))

    return np.array(out)








        












