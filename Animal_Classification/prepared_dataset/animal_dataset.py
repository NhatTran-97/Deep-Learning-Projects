import os
import torch
from torch.utils.data import Dataset,DataLoader

from torchvision.transforms import ToTensor, Resize, Compose, RandomAffine, ColorJitter
import cv2
import numpy as np
from PIL import Image

from argparse import ArgumentParser

class Animal_Dataset(Dataset):
    def __init__(self,root, train=True,transform = None):

        if train:
            mode = "train"
        else:
            mode = "test"

        root = os.path.join(root, mode)

        self.transform = transform

        self.categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

        self.images_path = []; self.labels = []

        for i , category in enumerate(self.categories):
            data_file_path = os.path.join(root, category); # save path not save values of images
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path, file_name) 
                self.images_path.append(file_path); 
                self.labels.append(i)
        
    def __len__(self):
        return  len(self.labels) 

    def __getitem__(self, idx):
        image_path = self.images_path[idx]; image = Image.open(image_path).convert("RGB")
        #image_path = self.images_path[idx]; image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_args():
    parser = ArgumentParser(description="CNN Training")
    parser.add_argument("--root","-r", type=str, default="dataset/animals", help="Root of the dataset")
    parser.add_argument("--epochs","-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size","-b", type=int, default=8, help="Batch Size")
    parser.add_argument("--image_size", "-i",type=int, default=224, help="Batch Size")

    args= parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    train_transform = Compose([  
        RandomAffine(
            degrees = (-5,5),
            translate = (0.05, 0.05),
            scale=(0.85, 1.15),
            shear = 5),

        ColorJitter(
            brightness = 0.5,
            contrast = 0.5,
            saturation = 0.25,
            hue = 0.2),
        Resize((args.image_size,args.image_size)),
        ToTensor()
    ])
    val_transform = Compose([
        Resize((args.image_size,args.image_size)),
        ToTensor()
    ])
    dataset_train = Animal_Dataset(root=args.root, train=True,transform = train_transform)
    dataset_val = Animal_Dataset(root=args.root, train=False,transform = val_transform)

    
    
    image, label = dataset_train.__getitem__(1432) # lấy 1 sample ra
    print(image.shape)
    image = (torch.permute(image, (1,2,0)) * 255.).numpy().astype(np.uint8) ; print(image.shape)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imshow("test image", image)
    cv2.waitKey(0)

    exit(0)


    train_dataloader = DataLoader(dataset = dataset_train, 
                                    batch_size=args.batch_size , 
                                    num_workers=4, 
                                    shuffle=True,
                                    drop_last = False,)
    
    val_dataloader = DataLoader(dataset = dataset_val, 
                                batch_size = args.batch_size, 
                                num_workers=4, 
                                shuffle=False,
                                drop_last = False,)