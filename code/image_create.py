import numpy as np
import matplotlib.pyplot as plt
from utils import save_image

input=np.load('../data/cylinder_u.npy')
output=np.load('../output/980.npy')

img_list=[0,100,200,300]

save_image(input,output,img_list)