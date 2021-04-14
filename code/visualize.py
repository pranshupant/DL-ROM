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

def MSE_barplot():

    mse_values = []
    xticks = []

    datasets= ['2d_cylinder', 'boussinesq', '2d_cylinder_cfd','SST','channel_flow']

    for i in datasets:
        try:
            mse=np.load(f'../results/{i}/MSE.npy')
            print(mse)
            mse_values.append(-1*np.log(mse))
            xticks.append(i)
        except:
            print(f'MSE for {i} not found')
    
    N = len(mse_values)            #number of the bars 
    ind = np.arange(N)  # the x locations for the groups

    fig = plt.figure(figsize = [4,3], dpi = 600)
    width = 0.35
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(width)
    tick_width = 0.35
    plt.tick_params(direction = 'in', width = tick_width)

    rects1 = ax.bar(ind, mse_values,width,color='blue',error_kw=dict(lw=1),capsize=2)
    ww = 0.16
    ax.set_ylabel('MSE per pixel', fontsize=10)
    plt.xticks(ind,xticks,rotation=0, fontsize = 8)
    plt.yticks(fontsize = 12)
    plt.xlabel('DL-ROM', fontsize=10)
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.003*h, '%.2f'%float(h),
                    ha='center', va='bottom',fontsize=8)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    plt.tick_params(direction = 'in', width = 1.5)

    autolabel(rects1)

    plt.show()
    filename = '../results/MSE_barplot.png'
    fig.savefig(filename, bbox_inches = 'tight')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d_set', dest='dset', type=str, default='2d_cylinder', help="Name of Dataset")
    parser.add_argument('-freq', dest='freq', type=int, default=20, help="Frequency for saving plots")
    parser.add_argument('--MSE', dest='barplot', action='store_true')

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

    if args.barplot:
        MSE_barplot()








