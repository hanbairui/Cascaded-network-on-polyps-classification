import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from torch.utils.data import ConcatDataset
import random
from sklearn.metrics import roc_auc_score
import numpy as np
from PIL import Image
from frr import FastReflectionRemoval
from torch.utils.data import Dataset
from typing import Sequence
import torchvision.transforms.functional as TF
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

## PICCOLO dataset contains 854*480 and 1920*1080 img
h1=480##image sizes according to dataset
h2=1080
cut1=430## crop the image to discard the black border, depending on image size and frames.
cut2=950

class CropCentralArea(object):#central area is selected as the region of interest, disgard the black region around the images
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        w, h = img.size
        if h==h1:
            left = (w - cut1) / 2
            top = (h - cut1) / 2
            right = (w + cut1) / 2
            bottom = (h + cut1) / 2

        if h==h2:
            left = (w - cut2) / 2
            top = (h - cut2) / 2
            right = (w + cut2) / 2
            bottom = (h + cut2) / 2

        return img.crop((left, top, right, bottom))

class Rotate90:

    def __init__(self, angles:Sequence[int]):

        self.angles = angles


    def __call__(self, x):

        angle = random.choice(self.angles)

        return TF.rotate(x, angle)
    
class FastReflectionRemovalTransform:
    def __init__(self, h):
        self.h = h
        self.alg = FastReflectionRemoval(h=h)

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Normalize pixel values to be in the range [0, 1]
        img_np = img_np / 255.0
        # print(img_np)
        # Run the algorithm and get the dereflected image
        dereflected_img_np = self.alg.remove_reflection(img_np)

        # Convert back to PIL image
        dereflected_img = Image.fromarray((dereflected_img_np * 255).astype(np.uint8))

        return dereflected_img

class Augmentation_per_class(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Get the image and label at the given index
        img, label = self.data[index]
        # if random.random() > 0.5:
        augmentation_transform = create_augmentation_transform()
        img = augmentation_transform(img)

        return img, label

    def __len__(self):

        return len(self.data)
    
    
def create_augmentation_transform():
    transform_augmentation = transforms.Compose([
        FastReflectionRemovalTransform(h=0.02),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        Rotate90([0, 0, 90, 180, 270, 360]),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.8),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0)], p=0.8),
        transforms.RandomChoice([
            transforms.RandomPerspective(distortion_scale=0.25),
            transforms.RandomAffine(degrees=(-45, 45), translate=(0, 0.0625), scale=(1.0, 1.05))
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_augmentation

#show image if you want to see images after augmentation
def show_augmented_img(data_loader):
    count=0
    try:

            batch, label= next(iter(data_loader))
            print(batch.shape)
            print(label[0])
            # Create a grid of images
            grid = vutils.make_grid(batch[:9], nrow=3, padding=2, normalize=True)

            # Convert the grid to a numpy array and transpose the dimensions
            grid = np.transpose(grid.numpy(), (1, 2, 0))
            # Show the grid
            plt.imshow(grid)
            plt.axis('off')
            plt.show()

    except BrokenPipeError:
        print("Broken pipe error occurred", file=sys.stderr)