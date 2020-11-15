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
        # print(param[1].requires_grad)

    return model