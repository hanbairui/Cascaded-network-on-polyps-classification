import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import geffnet
from torch.utils.data import ConcatDataset
import random
import joblib
from sklearn.metrics import roc_auc_score
import numpy as np
from PIL import Image
from frr import FastReflectionRemoval
from torch.utils.data import Dataset
from typing import Sequence
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from model import CustomResNet152,CustomResNet101,CustomEfficientnet_SVM,train_and_evaluate_model,extract_feature_maps,train_and_evaluate_svm_classifier,Resnet101,Resnet152
from evaluation import compute_metrics_AUC,compute_metrics
from Preprocessing import CropCentralArea,Rotate90,FastReflectionRemovalTransform,Augmentation_per_class,show_augmented_img

dir_data=os.path.abspath('D:\polyps\classify\PICCOLO') # directory with the images (root)
dir_train=os.path.join(dir_data, 'train','polyps_balance_1')##here load the train,validation set with 2 classes, with specific stage
dir_val=os.path.join(dir_data, 'validation', 'cascade1')
batch_size = 16
## PICCOLO dataset contains 854*480 and 1920*1080 img

transform__nomalize = transforms.Compose([
    CropCentralArea(height=512, width=512),
    transforms.Resize((256,256)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

##for 2nd stage with cropped images, no need to resize again
transform__nomalize_cropped = transforms.Compose([
    # CropCentralArea(height=512, width=512),
    # transforms.Resize((256,256)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform__nomalize_SVM = transforms.Compose([
    CropCentralArea(height=512, width=512),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



transform__nomalize_val = transforms.Compose([
    CropCentralArea(height=512, width=512),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



##make train dataset and validation set, so the dataloader has augmentation training data, with constant size 256*256, and validation set is only normalized and resized, without augmentation
train_ds_original = datasets.ImageFolder(
    root=dir_train,
    transform=transform__nomalize #transform__nomalize_cropped
)
train_in=train_ds_original


train_dl = torch.utils.data.DataLoader(
    Augmentation_per_class(train_in),
    batch_size=batch_size,
    shuffle=True,
    # num_workers=2,
    pin_memory=True
)

val_ds_original = datasets.ImageFolder(
    root=dir_val,
    transform=transform__nomalize_val
)##only normalize for validation

val_dl = torch.utils.data.DataLoader(
    val_ds_original,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=2,
    pin_memory=True
)

train_ds_original_SVM = datasets.ImageFolder(
    root=dir_train,
    transform=transform__nomalize_SVM
)

##show images after augmentation
show_augmented_img(data_loader=train_dl)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classify_mode='CNN'#or 'SVM'

#for CNN classifier
if classify_mode=='CNN':
    ##to ensure the same model structure for further loading, use function 'Resnet101','Resnet152'....; To make model structure more reasonable, use Class: 'CustomResNet101','CustomResNet152'...but the model trained on old version could not be loaded in new class. 
    model = Resnet101()#CustomResNet152()
    model.to(device)
    numpy_train_loss, numpy_val_loss, numpy_train_acc, numpy_val_acc = train_and_evaluate_model(model, train_dl, val_dl,batch_size=batch_size, num_epochs=1, class_weights=[1.0,1.0], lr=1e-4)

    #evaluation
    confusion_matrix, precision, recall, f1_score, macro_precision, macro_recall, macro_f1_score= compute_metrics(model, val_dl,device=device)
    print("Confusion matrix:\n", confusion_matrix)
    print("Precision:\n", precision)
    print("Recall:\n", recall)
    print("F1-score:\n", f1_score)
    print("Macro-averaged precision:", macro_precision)
    print("Macro-averaged recall:", macro_recall)
    print("Macro-averaged F1-score:", macro_f1_score)   

    ##save model
    model_save_name = 'resnet152FC_frozen_balance_cascade1_1.pt'
    save_path = f"./{model_save_name}"
    torch.save(model.state_dict(), save_path)

#for SVM classifier
if classify_mode=='SVM':
    model = CustomEfficientnet_SVM()
    model.to(device)
    accuracy,conf_matrix,macro_precision,macro_recall,macro_f1,svm_classifier,scaler = train_and_evaluate_svm_classifier(train_ds_original_SVM, val_ds_original, model, device='cuda')
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Macro F1:", macro_f1)

    ##save model
    model_path = './effb4_SVM_cas1.pkl'
    joblib.dump(svm_classifier, model_path)
    joblib.dump(scaler, './effb4_SVM_scaler.pkl')

    print(f"Trained SVM model saved at {model_path}")

