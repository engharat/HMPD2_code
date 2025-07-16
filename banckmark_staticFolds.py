import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
import json
from Utility.yamlManager import yamlManager
from torchBackend.utils import *
from torchBackend.DatasetManager import MicroplastDataset
from torchBackend.DatasetManager import dataLoaderGenerator
from torchBackend.trainTest import trainTest
import logging
torch.backends.cudnn.benchmark =  True
torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/costuomConf.yaml', help='configuration file')
    parser.add_argument('--name', type=str, default='test', help='banchmark_name')
    parser.add_argument('--dataset', type=str, help='dataset folder')
    parser.add_argument('--gt', type=str, help='dataset ground truth')
    parser.add_argument('--device', type=str, default='cuda', help='device option are: cpu, gpu, mps' )
    parser.add_argument('--save', type=str, default='False', help='If tru enable the model saving')
    opt = parser.parse_args()
    config_file = opt.config
    banckmark_name = opt.name
    dataset_path = opt.dataset
    gt_path = opt.gt
    if opt.device == 'mps':
        device = ("mps" if torch.backends.mps.is_available() else "cpu")
    elif opt.device == 'gpu':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif opt.device == 'cpu':
        device = 'cpu'
    else:
        device = 'cpu'
        logging.warning(f"The device {opt.device} in not in the available list, "
                        f"the processing has been redirected on CPU")

    if not os.path.exists(f"./tests/{banckmark_name}"):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(f"./tests/{banckmark_name}")

    yM = yamlManager(config_file, f"./tests/{banckmark_name}/banchmarkconfiguration.yaml")
    configuration = yM.read_conf()

    num_epochs = configuration['num_epochs']
    learning_rate = configuration['learning_rate']
    train_CNN = configuration['train_CNN']
    batch_size = configuration['batch_size']
    shuffle = configuration['shuffle']
    pin_memory = configuration['pin_memory']
    num_workers = configuration['num_workers']
    transform_resize = configuration['transform_resize']
    transform_crop = configuration['transform_crop']
    transform_normalize_mean = configuration['transform_normalize_mean']
    transform_normalize_var = configuration['transform_normalize_var']
    listofNetwork = configuration['listofNetwork']
    num_classes = configuration['num_classes']
    channel = configuration['channel']

    transform = getTransformer(transform_resize, transform_crop, transform_normalize_mean, transform_normalize_var)

    for k, m in listofNetwork.items():

        model = generateModel(k, num_classes)
        model.to(device)
        #model = torch.compile(model, backend="inductor")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        #import pdb; pdb.set_trace()

        reportData = []
        for fold in range(5):

            trainDataSet = MicroplastDataset(dataset_path, f"{gt_path}/train_{fold}.csv", transform=transform, channel=channel)
            testDataSet = MicroplastDataset(dataset_path, f"{gt_path}/test_{fold}.csv", transform=transform, channel=channel)

            train_loader = DataLoader(trainDataSet, batch_size=batch_size, shuffle=True)
            validation_loader = DataLoader(testDataSet, batch_size=batch_size, shuffle=True)

            print(f'FOLD {fold}')

            #train_loader, validation_loader = dataLoaderGenerator(dataset, train_ids, val_ids, batch_size)
            traintestfold = trainTest(model, device, criterion, optimizer, banckmark_name, k, fold, save=opt.save)
            traintestfold.train(train_loader, validation_loader, num_epochs)
            val_acc, conf, predictions, yGT, probs = traintestfold.check_full_accuracy(validation_loader)

            probs_1 = [tensor[1].item() for tensor in probs]
            yGT = [tensor.item() for tensor in yGT]
            predictions = [tensor.item() for tensor in predictions]

            current_data = {'fold': fold, 'gt': yGT, 'predictions': predictions, 'probsClass1': probs_1}

            reportData.append(current_data)
            traintestfold.reset_weights(model)

        with open(f"./tests/{banckmark_name}/{k}.json", "w") as final:
            json.dump(reportData, final)


