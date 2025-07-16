from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from PIL import Image
import os
import torch

def dataLoaderGenerator(dataset, train_ids, val_ids,batch_size):
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

    return train_loader, validation_loader


class MicroplastDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, channel='P'):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.channel = channel
        self.imgs = []
        if isinstance(self.channel,str):
            for index in range(0,len(self.annotations)):
                if (self.channel == "A") or (self.channel == "P") or (self.channel == "R"):
                    img_id = f"{self.annotations.iloc[index, 0]}_{self.channel}.bmp" 
                else:
                    img_id = f"{self.annotations.iloc[index, 0]}_{self.channel}.png"
                img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB").resize((256,256),Image.BILINEAR)
                self.imgs.append(img)
        if isinstance(self.channel,list):
            
            for index in range(0,len(self.annotations)):
                sample_img_list = []
                for elem in self.channel:
                    img_id = f"{self.annotations.iloc[index, 0]}_{elem}.png"
                    #import pdb; pdb.set_trace()
                    #conversion to RGB only needed if using a 3D model 
                    img = Image.open(os.path.join(self.root_dir, img_id)).convert('RGB').resize((256,256),Image.BILINEAR)
                    sample_img_list.append(img)
                self.imgs.append(sample_img_list)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        #import pdb; pdb.set_trace()
        img = self.imgs[index]
        if not isinstance(img,list):
            if self.transform is not None:
                img = self.transform(img)
            return img, y_label
        else:
            sample_img_list=[]
            for elem in img:
                if self.transform is not None:
                    elem = self.transform(elem) # img ha dimensione H x W ?
                sample_img_list.append(elem)
            #import pdb; pdb.set_trace()    
            sample_img_list = torch.stack(sample_img_list)  ##DA CONTROLLARE CON DEBUGGING, obiettivo: un tensore C x H x W
            #reshape only needed if we are using a VideoSwin Model 
            #sample_img_list = sample_img_list.reshape(3,12,224,224)
            sample_img_list = torch.squeeze(sample_img_list)
            #import pdb; pdb.set_trace()
            return sample_img_list, y_label


class MicroplastDataset_backup(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, channel='P'):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.channel = channel
        #import pdb; pdb.set_trace()
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = f"{self.annotations.iloc[index, 0]}_{self.channel}.bmp" if self.channel==('P'or'A'or'R') else f"{self.annotations.iloc[index, 0]}_{self.channel}.png"
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return img, y_label
