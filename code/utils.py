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

def load_transfer_learning_UNet_3D(pretrained, model, PATH, req_grad=True):

    checkpoint = torch.load(PATH)
    pretrained.load_state_dict(checkpoint, strict=False)

    layers = {}
    # print(model.named_parameters())

    for param in pretrained.named_parameters():
        if param[0][:2] != "d1" and param[0][:2] != "u5":
            # print(param[0])
            # print(type(param[1]))
            layers[param[0]] = param[1].data

    for param in model.named_parameters():
        if param[0] in layers.keys():
            param[1].data = layers[param[0]]
            param[1].requires_grad = req_grad


    # for param in model.named_parameters():
    #     print(param[0], param[1].requires_grad, param[1].data.shape)

    # for param in pretrained.named_parameters():
    #     print(param[0], param[1].requires_grad, param[1].data)

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

def save_loss(loss, dataset_name, mode='val'):
    np.save(f'../results/{dataset_name}/weights/{mode}_loss_dict.npy', loss)

def find_weight(dataset_name, test_epoch=None):

    if test_epoch is None:

        try:
            d = np.load(f'../results/{dataset_name}/weights/val_loss_dict.npy', allow_pickle=True).item()

            test_epoch = min(d.items(), key=lambda x: x[1])[0]
        except:
            test_epoch = 0

    PATH = f'../results/{dataset_name}/weights/{test_epoch}.pth'
    return PATH

def normalize_data(u):
    u = 2*(u-np.min(u))/(np.max(u)-np.min(u))-1
    return u

def MSE(dataset_name, pred, target):
    mse=np.sum((target-pred)**2)/(pred.shape[0]*pred.shape[1]*pred.shape[2])
    print(mse)
    np.save(f'../results/{dataset_name}/MSE.npy',mse)

def MSE_simulate(pred, target):
    mse=np.sum((target-pred)**2)/(pred.shape[0]*pred.shape[1])
    # print(mse)
    return mse

def plot_training(Train_Loss, Dev_Loss):
    # plt.title(f'Training & Validation Losses')
    fig, ax = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.yaxis.label.set_size(16)
    ax.xaxis.label.set_size(16)
    # ax.set_ylim([0.04, 0.16])
    plt.plot(Train_Loss, 'k-')
    plt.plot(Dev_Loss, 'g-')
    plt.legend(loc='best', labels=['Validation Loss', 'Training Loss' ])
    plt.savefig(f'training_plot.eps', dpi=600)
    plt.close()

def plot_training_from_dict(dataset_name = '2d_sq_cyl'):
    
    V = np.load(f'../results/{dataset_name}/weights/val_loss_dict_og.npy', allow_pickle=True).item()
    T = np.load(f'../results/{dataset_name}/weights/train_loss_dict_og.npy', allow_pickle=True).item()
    V_tl = np.load(f'../results/{dataset_name}/weights/val_loss_dict.npy', allow_pickle=True).item()
    T_tl = np.load(f'../results/{dataset_name}/weights/train_loss_dict.npy', allow_pickle=True).item()
    # T = np.array(T)
    # V = np.array(V)
    print(type(T))
    print(list(T.values()))
    fig, ax = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.yaxis.label.set_size(16)
    ax.xaxis.label.set_size(16)
    # ax.set_ylim([0.04, 0.16])
    plt.plot(list(V.keys()), list(V.values()), 'g-')
    plt.plot(list(T.keys()), list(T.values()), 'k-')    
    plt.plot(list(V_tl.keys()), list(V_tl.values()), 'b-')
    plt.plot(list(T_tl.keys()), list(T_tl.values()), 'r-')
    
    plt.legend(loc='best', labels=['Validation Loss', 'Training Loss', 'Validation Loss_TL', 'Training Loss_TL' ])
    # plt.show()
    
    plt.savefig(f'../results/{dataset_name}/training_plot_tl.eps', dpi=600)
    plt.close()

if __name__ == '__main__':

    input_dim = torch.randn(10, 320, 160)
    print(input_dim.shape)
    out = conv3D_shape(input_dim.shape, (3,8,4),stride=(1,4,2),padding=(0,2,1))
    print(out)