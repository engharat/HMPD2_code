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

def import_gt(dataset_path, gt_path):
    annotations = pd.read_csv(gt_path)
    heights = []
    widths = []
    areas = []
    ratios = []
    for item in annotations.iterrows():
        img_id = f"{item[1].patchids}_P.bmp"
        img = Image.open(os.path.join(dataset_path, img_id)).convert("RGB")
        heights.append(img.height)
        widths.append(img.width)
        areas.append(img.height*img.width)
        ratios.append(float(img.height)/float(img.width))
    annotations['heights'] = heights
    annotations['widths'] = widths
    annotations['areas'] = areas
    annotations['ratios'] = ratios
    return annotations



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset folder')
    parser.add_argument('--gt', type=str, help='dataset ground truth')
    parser.add_argument('--maxrangex', type=int, help='maximum bin range')
    parser.add_argument('--maxrangey', type=int, help='maximum y range')
    opt = parser.parse_args()
    dataset_path = opt.dataset
    gt_path = opt.gt
    maxrangex = opt.maxrangex
    maxrangey = opt.maxrangey



    annotations = import_gt(dataset_path, gt_path)

    stat = []
    for i in [0,1]:
        an = annotations[annotations.classes == i].areas.to_list()
        an_h = annotations[annotations.classes == i].heights.to_list()
        an_w = annotations[annotations.classes == i].widths.to_list()
        plt.figure(0).clf()
        plt.hist(an, bins=100)
        plt.xlim(xmin=0, xmax=maxrangex)
        plt.ylim(ymin=0, ymax=maxrangey)
        plt.grid(True)
        plt.xlabel("Patch area [pixels]")
        plt.ylabel("Occurrencies")
        #plt.title(f"Distribution of patche areas {i}")
        plt.savefig(f"./statistics/area_hist{i}")
        an = np.array(an)
        an_h = np.array(an_h)
        an_w = np.array(an_w)
        #stat.append({'label': i, 'min_area': an.min(), 'avg_area': f"{an.mean():.2f}", 'size': an.shape[0]})
        stat.append({'label': i, 'min_area': int(an.min()), 'max_area': int(an.max()), 'avg_area': f"{an.mean():.2f}",
                     'min_h': int(an_h.min()), 'max_h': int(an_h.max()),
                     'min_w': int(an_w.min()), 'max_w': int(an_w.max()),
                     'size': an.shape[0]})

    with open('./statistics/stat.json', 'w') as f:
        json.dump(stat, f)



