import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import models
import geffnet
import joblib
from Preprocessing import CropCentralArea,Rotate90,FastReflectionRemovalTransform,Augmentation_per_class,show_augmented_img
from model import CustomResNet152,CustomResNet101,CustomEfficientnet_SVM,train_and_evaluate_model,extract_feature_maps,train_and_evaluate_svm_classifier,Resnet101,Resnet152
from evaluation import compute_metrics_connection

dir_data=os.path.abspath('D:\polyps\classify\PICCOLO') # directory with the images (root)
dir_train=os.path.join(dir_data, 'train','polyps_balance')
dir_val=os.path.join(dir_data, 'validation', 'polyps')##here load the validation set with 3 classes
dir_test=os.path.join(dir_data, 'test', 'polyps')

batch_size = 1

transform__nomalize = transforms.Compose([
    CropCentralArea(height=512, width=512),
    transforms.Resize((256,256)),
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


net1 = Resnet101(device=device)
net2 = Resnet101(device=device)

net1.load_state_dict(torch.load('./resnet101FC_frozen_balance_cascade1_2.pt'))
net1.to(device)

net2.load_state_dict(torch.load('./resnet101FC_frozen_balance_cascade2_3.pt'))
net2.to(device)

net1.eval()
net2.eval()

cascade_model_ls=[net1,net2]
confusion_matrix, precision,recall,f1_score,macro_precision,macro_recall,macro_f1_score= compute_metrics_connection(cascade_model_ls,val_dl)

print("Confusion matrix:\n", confusion_matrix)
print("Precision:\n", precision)
print("Recall:\n", recall)
print("F1-score:\n", f1_score)
print("Macro-averaged precision:", macro_precision)
print("Macro-averaged recall:", macro_recall)
print("Macro-averaged F1-score:", macro_f1_score)