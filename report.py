import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, f1_score, recall_score
from glob import glob
import json
import argparse


class reporter():

    def __init__(self, experiment):
        self.folder = f"./tests/{experiment}"
        files = glob(f"{self.folder}/*.json")
        files.sort()
        self.networks = dict([(f.split('/')[-1][0:-5], f) for f in files])

    def checkAccuracy(self, predictions, yGT):
        num_correct = 0
        num_samples = 0
        num_correct = sum(a == b for a, b in zip(predictions, yGT))
        num_samples = len(predictions)

        #accuracy
        return float(num_correct) / float(num_samples) * 100

    def processDataFromJson(self):
        resutls = {}
        prob_1_allNetwork = []
        all_network_names = []
        all_yGT = []
        for n, f in self.networks.items():
            print(n)
            with open(f, 'r') as c:
                data = json.load(c)
            c.close()
            yGT = []
            probs_1 = []
            predictions = []
            for d in data:
                yGT.extend(d['gt'])
                probs_1.extend(d['probsClass1'])
                predictions.extend(d['predictions'])
            self.plot_sklearn_roc_curve(yGT, probs_1, n)
            resutls[n] = {"acc": self.checkAccuracy(predictions, yGT), "recall": recall_score(yGT, predictions, average='binary'), "precision": precision_score(yGT, predictions, average='binary'), "f1": f1_score(yGT, predictions, average='binary')}
            prob_1_allNetwork.append(probs_1)
            all_network_names.append(n)
            all_yGT.append(yGT)

            with open(f"{self.folder}/results.json",'w') as c:
                json.dump(resutls, c)
        self.overall_roc_curve(all_yGT, prob_1_allNetwork, all_network_names)





    def plot_sklearn_roc_curve(self, y_real, y_prob_1, network):
        RocCurveDisplay.from_predictions(
            y_real,
            y_prob_1,
            name="micro-average OvR",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{network} \nReceiver Operating Characteristic")
        plt.legend()
        plt.savefig(f"{self.folder}/{network}", dpi=400)

    def overall_roc_curve(self, list_of_y_real, list_of_y_prob_1, networks):

        plt.figure(0).clf()
        for net, y_real, y_prob_1 in zip(networks, list_of_y_real, list_of_y_prob_1):
            fpr, tpr, thresholds = roc_curve(y_real, y_prob_1, pos_label=1)
            auc = roc_auc_score(y_real, y_prob_1)
            plt.plot(fpr, tpr, label=f"{net}, AUC= {auc:.2f}")

        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        #plt.title(f"Overall ROC \nReceiver Operating Characteristic")
        #h, l = plt.gca().get_legend_handles_labels()
        #res = [h[i[0]] for i in sorted(enumerate(l), key=lambda x: x[1])]
        plt.legend()
        plt.savefig(f"{self.folder}/totalROC", dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='banchmark_name')
    opt = parser.parse_args()
    banckmark_name = opt.name

    a = reporter(banckmark_name)
    a.processDataFromJson()
