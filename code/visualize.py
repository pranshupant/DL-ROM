import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

'''
python visualize.py -d_set 2d_cylinder -freq 10
'''

def plot_results(pred, labels, dataset_name, freq):
    assert(pred.shape == labels.shape)

    for i in range(0, pred.shape[0], freq):
    
        plt.imshow(pred[i], cmap='RdBu')
        plt.axis('off')
        plt.savefig(f"../results/{dataset_name}/plots/{i}_pred.png", bbox_inches='tight', dpi=600)

        plt.imshow(labels[i], cmap= 'RdBu')
        plt.axis('off')
        plt.savefig(f"../results/{dataset_name}/plots/{i}_label.png", bbox_inches='tight',  dpi=600)

        plt.imshow(pred[i]-labels[i], cmap= 'gist_gray')
        plt.axis('off')
        plt.savefig(f"../results/{dataset_name}/plots/{i}_diff.png", bbox_inches='tight',  dpi=600)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d_set', dest='dset', type=str, default='2d_cylinder', help="Name of Dataset")
    parser.add_argument('-freq', dest='freq', type=int, default=20, help="Frequency for saving plots")

    args = parser.parse_args()
    dataset_name = args.dset
    freq = args.freq

    if not os.path.exists(f"../results/{dataset_name}"):
        os.mkdir(f"../results/{dataset_name}")

    if not os.path.exists(f"../results/{dataset_name}/plots"):
        os.mkdir(f"../results/{dataset_name}/plots")

    pred = np.load(f'../results/{dataset_name}/output/predictions.npy')
    labels = np.load(f'../results/{dataset_name}/output/labels.npy')

    # val_size, imageh, imagew
    plot_results(pred, labels, dataset_name, freq)








