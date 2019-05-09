import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from  PIL import Image
import json
import transforms
import nibabel as nib

from augmentation import *
from visualization import *

class ACDCDataset(Dataset):
    def __init__(self, data_path="../data/", data_type = "train", transform=None):
        #store some input 
        self.data_path = str(data_path)
        self.data_type = str(data_type)
        self.filename = data_path+"index/"+data_type+"_data.txt"
        self.transform = transform
        self.data = []

        #parse the txt to store the necessary information of output
        file = open(self.filename, 'r').readlines()
        for i in range(len(file)):
            entry = file[i].strip()
            self.data.append(entry)

    def __len__(self):
        #return the length of the data numbers
        return len(self.data)

    def __getitem__(self, idx):
        # parse info file
        patient_path = self.data_path+self.data_type+"/"+"patient"+'{:03d}'.format(int(self.data[idx]))
        patient_info = patient_path + "/info.cfg"
        # parse the txt to store the necessary information of patient
        file = open(patient_info, 'r').readlines()
        ED_frame = int(file[0].split(":")[1])
        ES_frame = int(file[1].split(":")[1])
        
        # path
        img_path_ED = patient_path +"/patient"+'{:03d}'.format(int(self.data[idx]))+"_frame{:02d}.nii.gz".format(ED_frame)
        img_path_ES = patient_path +"/patient"+'{:03d}'.format(int(self.data[idx]))+"_frame{:02d}.nii.gz".format(ES_frame)
        label_path_ED = patient_path +"/patient"+'{:03d}'.format(int(self.data[idx]))+"_frame{:02d}_gt.nii.gz".format(ED_frame)
        label_path_ES = patient_path +"/patient"+'{:03d}'.format(int(self.data[idx]))+"_frame{:02d}_gt.nii.gz".format(ES_frame)
#         print(img_path_ED)
#         print(img_path_ES)
#         print(label_path_ED)
#         print(label_path_ES)

        # get img from file
        img_ED = nib.load(img_path_ED).get_data()
        img_ES = nib.load(img_path_ES).get_data()
        img_ED = resize(img_ED,(256,256,10))
        img_ES = resize(img_ES,(256,256,10))
        
        # parse label
        label_ED = nib.load(label_path_ED).get_data()
        label_ES = nib.load(label_path_ES).get_data()
        label_ED = resize(label_ED,(256,256,10))
        label_ES = resize(label_ES,(256,256,10))
        
#         print(img_ED.shape)
#         print(img_ES.shape)
#         print(label_ED.shape)
#         print(label_ES.shape)
        
        # find mean and std
        mean_ED = np.mean(img_ED)
        std_ED = np.std(img_ED)
        mean_ES = np.mean(img_ES)
        std_ES = np.std(img_ES)
        
        # augment dataset
        if self.transform is not None:
            img_ED,label_ED = transforms.augment(img_ED,label_ED)
            img_ES,label_ES = transforms.augment(img_ES,label_ES)
        
        # apply normalization only to img
        img_ED = (img_ED - mean_ED)/std_ED
        img_ES = (img_ES - mean_ES)/std_ES
            
        img_ED = torch.from_numpy(img_ED).permute(2, 0, 1)
        img_ES = torch.from_numpy(img_ES).permute(2, 0, 1)
        label_ED = torch.from_numpy(label_ED).permute(2, 0, 1)
        label_ES = torch.from_numpy(label_ES).permute(2, 0, 1)
        
        return img_ED,img_ES,label_ED,label_ES

if __name__ == "__main__":
    dataset=ACDCDataset(transform=None)
    idx = 0
    dataset[idx]
    plt.imshow(dataset[idx][0].permute(1,2,0)[:,:,3],cmap='gray')
    plt.show()
    plt.imshow(dataset[idx][1].permute(1,2,0)[:,:,3],cmap='gray')
    plt.show()
    plt.imshow(dataset[idx][2].permute(1,2,0)[:,:,3],cmap='gray')
    plt.show()
    plt.imshow(dataset[idx][3].permute(1,2,0)[:,:,3],cmap='gray')
    plt.show()