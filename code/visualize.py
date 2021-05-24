import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from dataset_create import createAnimation

'''
python visualize.py -mode results -d_set 2d_cylinder -freq 10
python visualize.py -mode simulate -d_set 2d_cylinder
'''

def plot_results(pred, labels, mode, dataset_name, freq):
    assert(pred.shape == labels.shape)

    for i in range(0, pred.shape[0], freq):
        
        print(f'Plotted {i} / {pred.shape[0]}')
        plt.imshow(pred[i], cmap='RdBu', vmin=-1., vmax=1)
        plt.axis('off')
        plt.savefig(f"../{mode}/{dataset_name}/plots/{i}_pred.png", bbox_inches='tight', dpi=600)
        plt.close()

        plt.imshow(labels[i], cmap= 'RdBu', vmin=-1., vmax=1)
        plt.axis('off')
        plt.savefig(f"../{mode}/{dataset_name}/plots/{i}_label.png", bbox_inches='tight',  dpi=600)
        plt.close()

        plt.imshow(pred[i]-labels[i], cmap= 'bwr', vmin=-1., vmax=1)
        plt.axis('off')
        plt.savefig(f"../{mode}/{dataset_name}/plots/{i}_diff.png", bbox_inches='tight',  dpi=600)
        plt.close()

def plot_simulate(mse): 
    plt.plot(-1*np.log(mse), "ko-")
    plt.xlabel("Epoch Number")
    plt.ylabel("Negative Log. MSE (per pixel)")

    plt.savefig(f"../{mode}/{dataset_name}/plots/mse_lineplot.png", bbox_inches='tight',  dpi=600)
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d_set', dest='dset', type=str, default='2d_cylinder', help="Name of Dataset")
    parser.add_argument('-freq', dest='freq', type=int, default=20, help="Frequency for saving plots")
    parser.add_argument('-mode', dest='mode', type=str, default='results', help="result/simulate")

    args = parser.parse_args()
    dataset_name = args.dset
    freq = args.freq
    mode = args.mode

    if not os.path.exists(f"../{mode}/{dataset_name}"):
        os.mkdir(f"../{mode}/{dataset_name}")

    if not os.path.exists(f"../{mode}/{dataset_name}/plots"):
        os.mkdir(f"../{mode}/{dataset_name}/plots")

    # val_size, imageh, imagew
    if mode == "results":
        pred = np.load(f'../{mode}/{dataset_name}/output/predictions.npy')
        labels = np.load(f'../{mode}/{dataset_name}/output/labels.npy')
        plot_results(pred, labels, mode, dataset_name, freq)

    elif mode == "simulate":
        pred = np.load(f'../{mode}/{dataset_name}/predictions.npy')
        labels = np.load(f'../{mode}/{dataset_name}/labels.npy')
        mse = np.load(f'../{mode}/{dataset_name}/mse.npy')
        # plot_results(pred, labels, mode, dataset_name, freq)
        plot_simulate(mse)
        createAnimation(pred, dataset_name + "_pred")
        createAnimation(labels, dataset_name + "_ground_truth")
    else:
        print("Incorrect Mode!")