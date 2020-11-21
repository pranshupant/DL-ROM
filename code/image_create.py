import numpy as np
import matplotlib.pyplot as plt
from utils import save_image, save_image_lstm
import os

if __name__ == '__main__':

    LSTM = True

    if not os.path.exists("../Images"):
        os.mkdir("../Images")

    # i=np.load('../data/cylinder_u.npy')
    i=np.load('../data/boussinesq_u.npy')
    output=np.load('../output/140.npy')

    img_list=[100,200, 600, 900, 1200]

    if not LSTM:        

        save_image(i,output,img_list)

    else:

        save_image_lstm(i,output,img_list)

