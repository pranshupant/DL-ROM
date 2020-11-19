import numpy as np
import matplotlib.pyplot as plt
from utils import save_image
import os

if __name__ == '__main__':

    if not os.path.exists("../Images"):
        os.mkdir("../Images")

    i=np.load('../data/boussinesq_u.npy')
    output=np.load('../output/bous_500.npy')
    output=output.reshape(-1,450,150)
    print(output.shape)

    img_list=[0,500,1000,1500,2000]

    save_image(i,output,img_list)