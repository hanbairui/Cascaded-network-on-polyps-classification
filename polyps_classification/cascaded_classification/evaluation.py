import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics_AUC(model, dataloader,device):
    n_classes = len(dataloader.dataset.classes)
    n_examples = len(dataloader.dataset)
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

            # Collect labels and probabilities for AUC calculation
            all_labels.append(labels.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probabilities = np.concatenate(all_probabilities)

    # Calculate AUC
    auc_scores = []
    for class_idx in range(n_classes-1):
        class_labels = (all_labels == class_idx).astype(np.int32)
        class_probabilities = all_probabilities[:, class_idx]
        auc = roc_auc_score(class_labels, class_probabilities)

    return auc

def compute_metrics(model, dataloader,device):
    n_classes = len(dataloader.dataset.classes)
    n_examples = len(dataloader.dataset)
    confusion_matrix = torch.zeros(n_classes, n_classes)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print(outputs)
            # break
            probabilities = torch.softmax(outputs, dim=1)##use softmax to avoid negative value for nn.CrossEntropyLoss, which expected to get positive logits
            preds = torch.argmax(probabilities, dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


    precision = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, dim=0)
    recall = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, dim=1)
    f1_score = 2 * precision * recall / (precision + recall)
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1_score = f1_score.mean().item()


    return confusion_matrix, precision, recall, f1_score, macro_precision, macro_recall, macro_f1_score

##for CNN models to evaluation connected network, no SVM in each stages
def compute_metrics_connection(cascade_models, dataloader,device='cuda'):
    n_classes = len(dataloader.dataset.classes)
    confusion_matrix = torch.zeros(n_classes, n_classes)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            final_predicted_labels = labels.clone().unsqueeze(1)  # Initialize final predicted labels with true labels

            for j, model in enumerate(cascade_models):
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probabilities, dim=1)

                if j == 0:  # First classifier
                    final_predicted_labels = torch.where((preds == 1) |  (preds == 0), preds.unsqueeze(1), final_predicted_labels)
                    # final_predicted_labels = torch.where( (preds == 1), preds.unsqueeze(1), final_predicted_labels)
                   
                else:  # Subsequent classifiers
                    # non_class0_preds = (preds != 0).long()
                    final_predicted_labels = torch.where(final_predicted_labels == 1, preds.unsqueeze(1) + j, final_predicted_labels)

            for t, p in zip(labels.view(-1), final_predicted_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    precision = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, dim=0)
    recall = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, dim=1)
    f1_score = 2 * precision * recall / (precision + recall)
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1_score = f1_score.mean().item()

    return (
        confusion_matrix,
        precision,
        recall,
        f1_score,
        macro_precision,
        macro_recall,
        macro_f1_score,

    )