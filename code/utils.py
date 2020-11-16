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
    