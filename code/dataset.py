import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from  PIL import Image
import json
import nibabel as nib

from augmentation import *
from albumentations import *
from visualization import *

class ACDCDataset(Dataset):
    def __init__(self, data_path="../data/", data_type = "train", transform_both=None, transform_image=None):
        #store some input 
        self.data_path = str(data_path)
        self.data_type = str(data_type)
        self.filename = data_path+"index/"+data_type+"_data.txt"
        self.transform_both = transform_both
        self.transform_image = transform_image
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
        if self.data_type=="validation":
            self.data_type="train" # train and validation share the same folder
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
#         print(img_ED.dtype)
        img_ED = pad_nd_image(img_ED,(256,256,10))
        img_ES = pad_nd_image(img_ES,(256,256,10))
        
        # parse label
        label_ED = nib.load(label_path_ED).get_data()
        label_ES = nib.load(label_path_ES).get_data()        
        label_ED = pad_nd_image(label_ED,(256,256,10))
        label_ES = pad_nd_image(label_ES,(256,256,10))
        
#         print(img_ED.shape)
#         print(img_ES.shape)
#         print(label_ED.shape)
#         print(label_ES.shape)
                
        # augment dataset
        if self.transform_both is not None:
            augmented_ED = self.transform_both(image=img_ED,mask=label_ED)
            img_ED = augmented_ED['image']
            label_ED = augmented_ED['mask']
            augmented_ES = self.transform_both(image=img_ES,mask=label_ES)
            img_ES = augmented_ES['image']
            label_ES = augmented_ES['mask']
        if self.transform_image is not None:
            augmented_ED = self.transform_image(image=img_ED)
            img_ED = augmented_ED['image']
            augmented_ES = self.transform_image(image=img_ES)
            img_ES = augmented_ES['image']
            
        img_ED = torch.from_numpy(img_ED).permute(2, 0, 1)
        img_ES = torch.from_numpy(img_ES).permute(2, 0, 1)
        label_ED = torch.from_numpy(label_ED).permute(2, 0, 1)
        label_ES = torch.from_numpy(label_ES).permute(2, 0, 1)
        
        return img_ED,img_ES,label_ED,label_ES
    
class ACDCDataset_test(ACDCDataset):
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

        # get img from file
        img_ED = nib.load(img_path_ED).get_data()
        img_ES = nib.load(img_path_ES).get_data()
#         print(img_ED.dtype)
        img_ED = pad_nd_image(img_ED,(256,256,10))
        img_ES = pad_nd_image(img_ES,(256,256,10))

        if self.transform_image is not None:
            augmented_ED = self.transform_image(image=img_ED)
            img_ED = augmented_ED['image']
            augmented_ES = self.transform_image(image=img_ES)
            img_ES = augmented_ES['image']

            
        img_ED = torch.from_numpy(img_ED).permute(2, 0, 1)
        img_ES = torch.from_numpy(img_ES).permute(2, 0, 1)
        
        return img_ED,img_ES

if __name__ == "__main__":
    train_both_aug = Compose([
        RandomCrop(height=256, width=256, p=1)
    ])
    train_img_aug = Compose([
        Normalize(p=1,mean=np.array([0.5,]),std=np.array([0.5,])),
#         RandomBrightnessContrast(brightness_limit=1.0, contrast_limit=1.0,p=1.0),
#         RandomGamma(p=1)
#         Cutout(p=1)
#         ShiftScaleRotate(p=1)
        
    ])
    
    
    val_both_aug = Compose([
        RandomCrop(height=256, width=256, p=1)
    ])
    val_img_aug = Compose([
        Normalize(p=1,mean=np.array([0.5,]),std=np.array([0.5,]))
    ])
    
    test_img_aug = Compose([
        RandomCrop(height=256, width=256, p=1),
        Normalize(p=1,mean=np.array([0.5,]),std=np.array([0.5,]))
    ])
    
    dataset=ACDCDataset(data_type="train",transform_both=train_both_aug,transform_image=train_img_aug)
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
    
    print(dataset[idx][0].shape)
    print(dataset[idx][1].shape)
    print(dataset[idx][2].shape)
    print(dataset[idx][3].shape)
    
    dataset=ACDCDataset(data_type="validation",transform_both=val_both_aug,transform_image=val_img_aug)
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
    
    print(dataset[idx][0].shape)
    print(dataset[idx][1].shape)
    print(dataset[idx][2].shape)
    print(dataset[idx][3].shape)

    dataset=ACDCDataset_test(data_type="test",transform_image=test_img_aug)
    idx = 0
    dataset[idx]
    imshow(dataset[idx][0].permute(1,2,0)[:,:,3])
    imshow(dataset[idx][1].permute(1,2,0)[:,:,3])
    print(dataset[idx][0].shape)
    print(dataset[idx][1].shape)
        
