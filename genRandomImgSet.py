import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
from glob import glob
import json
import argparse
import pandas as pd
from PIL import Image
import os
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

def showImagesHorizontally(list_of_files):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image,cmap='Greys_r')
        axis('off')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset folder')
    parser.add_argument('--gt', type=str, help='dataset ground truth')
    parser.add_argument('--label', type=int, help='desired class for example generation')
    opt = parser.parse_args()
    dataset_path = opt.dataset
    gt_path = opt.gt
    i = opt.label

    df = pd.read_csv(gt_path)
    df_0 = df[df.classes == 0]
    df_1 = df[df.classes == 1]

    number_of_samples = 6
    df_0 = df_0.sample(number_of_samples)
    df_1 = df_1.sample(number_of_samples)

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(8., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )


    imagesList = []
    imgR = []
    imgA = []
    imgP = []

    for item in eval(f"df_{i}.iterrows()"):
        imgR.append(Image.open(os.path.join(dataset_path, f"{item[1].patchids}_R.bmp")).convert("RGB"))
        imgA.append(Image.open(os.path.join(dataset_path, f"{item[1].patchids}_A.bmp")).convert("RGB"))
        imgP.append(Image.open(os.path.join(dataset_path, f"{item[1].patchids}_P.bmp")).convert("RGB"))
    imagesList.extend(imgR)
    imagesList.extend(imgA)
    imagesList.extend(imgP)

    for ax, im in zip(grid, imagesList):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')

    plt.savefig(f"./statistics/example{i}.png")

