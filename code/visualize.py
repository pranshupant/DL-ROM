import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='dset', type=str, default='2d_cylinder', help="Name of Dataset")
    parser.add_argument(dest='epoch', type=str, default='20', help="Epoch for output")


    args = parser.parse_args()
    dataset_name = args.dset
    epoch =  args.epoch

    if not os.path.exists(f"../results/{dataset_name}/plots"):
        os.mkdir(f"../results/{dataset_name}/plots")

    pred = np.load(f'../results/{dataset_name}/output/{epoch}.npy')

    






