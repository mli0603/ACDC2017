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
from albumentations.pytorch import *
from visualization import *

image_size = (256,256)

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
        patient_info = patient_path + "/Info.cfg"
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
#         img_ED = pad_nd_image(img_ED,image_size)
#         img_ES = pad_nd_image(img_ES,image_size)
        # normalize
        img_ED = img_ED/np.max(img_ED)
        img_ES = img_ES/np.max(img_ES)

        # parse label
        label_ED = nib.load(label_path_ED).get_data()
        label_ES = nib.load(label_path_ES).get_data()        
#         label_ED = pad_nd_image(label_ED,image_size)
#         label_ES = pad_nd_image(label_ES,image_size)
        
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
        
        # clip to 0 to 1
        img_ED = np.clip(img_ED, 0, 1)
        img_ES = np.clip(img_ES, 0, 1)
        
        # zero-centre
        img_ED = (img_ED-0.5)/0.5
        img_ES = (img_ES-0.5)/0.5
        
        img_ED = torch.from_numpy(img_ED).permute(2, 0, 1).float()
        img_ES = torch.from_numpy(img_ES).permute(2, 0, 1).float()
        img = torch.cat((img_ED,img_ES),0)
#         print(torch.max(img_ED))
#         print(torch.min(img_ED))
        label_ED = torch.from_numpy(label_ED).permute(2, 0, 1)
        label_ES = torch.from_numpy(label_ES).permute(2, 0, 1)
        label = torch.cat((label_ED,label_ES),0)
#         print(torch.max(label_ED))
#         print(torch.min(label_ES))

        sample = {'img':img,'label':label, 'indx':idx}
        
        return sample

if __name__ == "__main__":
    train_both_aug = Compose([
        PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0,p=1),
        RandomCrop(height=256, width=256, p=1),
        Cutout(num_holes=8,p=0.5),
        OneOf([
            ShiftScaleRotate(p=0.6),
            HorizontalFlip(p=0.8),
            VerticalFlip(p=0.8)
        ])
    ])
    train_img_aug = Compose([
        OneOf([
            RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.5,0.5),p=0.9),
            RandomGamma(gamma_limit=(50,200),p=0.8)
        ]),
    ])
    
    
    val_both_aug = Compose([
        PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0,p=1),
        RandomCrop(height=256, width=256, p=1)
    ])

    
    train_dataset=ACDCDataset(data_type="train",transform_both=train_both_aug,transform_image=train_img_aug)
    train_generator = DataLoader(train_dataset,shuffle=False,batch_size=1,num_workers=8)

    for i_batch, sample_batch in enumerate(train_generator):
        img = sample_batch['img'].permute(1,0,2,3) # change the order so that we have num of image at the first place
        label = sample_batch['label'].permute(1,0,2,3) # change the order so that we have num of image at the first place
        
        print(img.shape)
        print(label.shape)
        
        imshow(img[0,:,:,:].permute(1,2,0),denormalize=False)
        imshow(label[0,:,:,:].permute(1,2,0),denormalize=False)
        imshow(img[-1,:,:,:].permute(1,2,0),denormalize=False)
        imshow(label[-1,:,:,:].permute(1,2,0),denormalize=False)
        break
        
        
    validation_dataset=ACDCDataset(data_type="validation",transform_both=val_both_aug,transform_image=None)
    validation_generator = DataLoader(validation_dataset,shuffle=False,batch_size=1,num_workers=8)

    for i_batch, sample_batch in enumerate(validation_generator):
        img = sample_batch['img'].permute(1,0,2,3) # change the order so that we have num of image at the first place
        label = sample_batch['label'].permute(1,0,2,3) # change the order so that we have num of image at the first place
        
        print(img.shape)
        print(label.shape)
        
        imshow(img[0,:,:,:].permute(1,2,0),denormalize=False)
        imshow(label[0,:,:,:].permute(1,2,0),denormalize=False)
        imshow(img[-1,:,:,:].permute(1,2,0),denormalize=False)
        imshow(label[-1,:,:,:].permute(1,2,0),denormalize=False)
        break
        
#     test_dataset=ACDCDataset(data_type="test",transform_both=None,transform_image=test_img_aug)
#     test_generator = DataLoader(validation_dataset,shuffle=True,batch_size=1,num_workers=8)

#     for i_batch, sample_batch in enumerate(test_generator):
#         img_ED = sample_batch['ED']
#         img_ES = sample_batch['ES']
        
#         imshow(img_ED[0,:,:,:].permute(1,2,0)[:,:,3],denormalize=False)
#         imshow(img_ES[0,:,:,:].permute(1,2,0)[:,:,3],denormalize=False)
#         break
        
