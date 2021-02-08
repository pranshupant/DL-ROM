import numpy as np
import torch 
import  matplotlib.pyplot as plt

def to_img(x):
    x=0.5*(x+1)
    x=x.clamp(0,1)
    x=x.view(x.size(0),1,80,640)
    return x

def save_image(input,output,img_no_list):
    '''
    imput: np.array
    output: np.array
    img_no_list: a list of image_nos between 0 to 301 [0,301)
    '''
    count=1
    for i in img_no_list:
        fig=plt.figure(count)
        plt.subplot(2, 1, 1)
        plt.imshow(input[i])
        plt.subplot(2,1,2)
        plt.imshow(output[i])
        name='../Images/'+str(i)+'.png'
        plt.savefig(name)
        count+=1

def latent_data(latent):

    data=np.load(latent,allow_pickle=True)

    temp=[]
    for i in range(data.shape[0]):
        temp.append(data[i,0])
    
    temp=np.array(temp)
    X=[]
    Y=[]
    for i in range(0,temp.shape[0]-25):
        x_temp=np.zeros((5,10))
        y_temp=np.zeros((5,10))
        for j in range(5):
            x_temp[j,:]=temp[i+5*j]
            y_temp[j,:]=temp[i+5*(j+1)]
        X.append(x_temp)
        Y.append(y_temp)
    X=np.array(X)
    Y=np.array(Y)
    return X,Y


def save_image_lstm(input,output,img_no_list):
    '''
    imput: np.array
    output: np.array
    img_no_list: a list of image_nos between 0 to 301 [0,301)
    '''
    count=1
    for i in img_no_list:
        fig=plt.figure(count)
        plt.subplot(3, 1, 1)
        plt.imshow(input[i])
        plt.subplot(3, 1, 2)
        plt.imshow(output[i])
        plt.subplot(3, 1, 3)
        plt.imshow(input[i+100])
        name='../Images/lstm_b_'+str(i)+'.png'
        plt.savefig(name)
        count+=1

def load_transfer_learning(pretrained, model, PATH):

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint, strict=False)

    layers = []

    for param in pretrained.named_parameters():
        layers.append(param[0])

    for param in model.named_parameters():
        if param[0] in layers:
            param[1].requires_grad = False

    # for param in model.named_parameters():
    #     print(param[0], param[1].requires_grad)

    return model

def insert_time_channel(data, channels):
    data_new = np.reshape(data[:(int(1501/channels)*channels),...],
        (-1, channels, data.shape[1], data.shape[2]))
    print('Time Channel Inserted')
    return data_new

def load_transfer_learning_TF(pretrained, model, PATH):

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint, strict=False)

    # layers = []

    # for param in pretrained.named_parameters():
    #     layers.append(param[0])

    # for param in model.named_parameters():
    #     if param[0] in layers:
    #         param[1].requires_grad = False

    # for param in model.named_parameters():
    #     print(param[0], param[1].requires_grad)

    return model