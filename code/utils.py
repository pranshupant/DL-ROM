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

def conv3D_shape(input_shape, kernel, stride, padding=(0,0,0), dilation=(1,1,1)):
    '''
    Function to help in convolution shape calculations. Eg. print(conv2D_shape((1024, 2048), (4,4), (2,4), (1,1), (1,1)))
    '''
    d_in, h_in, w_in = input_shape
    d_out = np.floor(((d_in + 2*padding[0]-dilation[0]*(kernel[0]-1)-1)/stride[0]) + 1)
    h_out = np.floor(((h_in + 2*padding[1]-dilation[1]*(kernel[1]-1)-1)/stride[1]) + 1)
    w_out = np.floor(((w_in + 2*padding[2]-dilation[2]*(kernel[2]-1)-1)/stride[2]) + 1)
    

    return (d_out, h_out, w_out)

def save_loss(loss, dataset_name):
    np.save(f'../results/{dataset_name}/weights/val_loss_dict.npy', loss)

def find_weight(dataset_name):

    d = np.load(f'../results/{dataset_name}/weights/val_loss_dict.npy', allow_pickle=True).item()

    test_epoch = min(d.items(), key=lambda x: x[1])[0]

    PATH = f'../results/{dataset_name}/weights/{test_epoch}.pth'
    return PATH

if __name__ == '__main__':

    input_dim = torch.randn(10, 450, 150)
    print(input_dim.shape)
    out = conv3D_shape(input_shape=input_dim.shape,kernel=(3,5,3),stride=(1,3,1),padding=(1,2,1))
    print(out)