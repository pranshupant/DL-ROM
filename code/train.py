import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision
from utils import to_img 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def training(model,train_loader,criterion,optimizer):
    model.train()
    avg_loss=[]

    for batch_num, (feats, labels) in enumerate(train_loader):
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

    print('Train Loss: {:.4f}'.format(sum(avg_loss)/len(avg_loss)))


def validation(model,test_loader,criterion):
    model.eval()
    avg_loss=[]
    out=[]
    
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)

        outputs=model(feats)
        temp=outputs[0].detach().cpu().numpy()
        out.append(temp.reshape(80,640))
        loss=criterion(outputs,labels)
        avg_loss.append(loss.item())
        del feats
        del labels
    model.train()
    print('Validation Loss: {:.4f}'.format(sum(avg_loss)/len(avg_loss)))

    return np.array(out)








        












