from turtle import forward
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import geffnet
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize


class CustomResNet152(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet152, self).__init__()

        # Load the pre-trained ResNet152 model
        self.resnet152 = models.resnet152(pretrained=True)

        # Freeze the parameters of the ResNet152 model
        for param in self.resnet152.parameters():
            param.requires_grad = False

        self.resnet152 = nn.Sequential(*list(self.resnet152.children())[:-1])

        # Custom fully connected layers
        self.custom_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # Pass the input through the ResNet152 model
        x = self.resnet152(x)

        # Pass the output through the custom fully connected layers
        x = self.custom_fc(x)

        return x

class CustomResNet101(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet101, self).__init__()

        # Load the pre-trained ResNet101 model
        self.net101 = models.resnet101(pretrained=True)

        for param in self.net101.parameters():
            param.requires_grad = False

        self.resnet101 = nn.Sequential(*list(self.resnet101.children())[:-1])

        # Custom fully connected layers
        self.custom_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # Pass the input through the ResNet152 model
        x = self.resnet101(x)

        # Pass the output through the custom fully connected layers
        x = self.custom_fc(x)

        return x
    
    def load_model(self, model_path):
        # Load the model's state dictionary
        state_dict = torch.load(model_path)
        # Load the state dictionary into the model
        self.load_state_dict(state_dict)




def train_and_evaluate_model(model, train_dl, val_dl, batch_size,num_epochs, class_weights, lr,device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create empty lists for storing metrics during training
    numpy_train_loss = []
    numpy_val_loss = []
    numpy_train_acc = []
    numpy_val_acc = []

    for epoch in range(num_epochs):
        # Train the model for one epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            n=10
            if i % n == n-1:
                print('[Epoch %d, Batch %d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / (n*batch_size)))
                running_loss = 0.0

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = epoch_loss / (i*batch_size)
        numpy_train_loss.append(epoch_loss)
        print('Epoch %d Train_loss: %.3f' % (epoch + 1, epoch_loss))

        train_loss = running_loss / (len(train_dl)*batch_size)
        train_acc = correct / total
        numpy_train_acc.append(train_acc)

        # Evaluate the model on the validation set
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dl):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss = running_loss / (len(val_dl)*batch_size)
        valid_acc = correct / total
        numpy_val_loss.append(valid_loss)
        numpy_val_acc.append(valid_acc)

        # Print epoch statistics
        print("Epoch {}/{} - Train Loss: {:.4f} - Train Acc: {:.4f} - Valid Loss: {:.4f} - Valid Acc: {:.4f}"
              .format(epoch + 1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))

    return numpy_train_loss, numpy_val_loss, numpy_train_acc, numpy_val_acc

# You can call the function like this:
# numpy_train_loss, numpy_val_loss, numpy_train_acc, numpy_val_acc = train_and_evaluate_model(model, train_dl, val_dl, num_epochs, class_weights, device)





##This net is used for SVM classifier
class CustomEfficientnet_SVM(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomEfficientnet_SVM, self).__init__()

        # Load the pre-trained ResNet101 model
        self.net =geffnet.create_model('tf_efficientnet_b4', pretrained=True)

        for param in self.net.parameters():
            param.requires_grad = False

        self.net = nn.Sequential(*list(self.net.children())[:-2])

    def forward(self, x):
        # Pass the input through the ResNet101 model
        x = self.net(x)

        return x

##This function is used to extract feature maps from encoders,and then feed to SVM 
def extract_feature_maps(model, input_data,device):
    with torch.no_grad():
        input_data = input_data.unsqueeze(0).to(device)
        feature_map = model(input_data)
        feature_map = torch.sum(feature_map, dim=(-1, -2))
        feature_map = feature_map.view(1, -1)
    return feature_map

def train_and_evaluate_svm_classifier(train_ds_original, val_ds_original, encoder, device='cuda'):
    # Train the SVM classifier on the training dataset(no augmentation in SVM)
    svm_vectors = []
    encoder.eval()
    for data, _ in train_ds_original:
        feature_map = extract_feature_maps(encoder, data,device=device)
        svm_vectors.append(feature_map)

    svm_vectors_concatenated = torch.cat(svm_vectors, dim=0)
    svm_vectors_concatenated = svm_vectors_concatenated.to(device)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(svm_vectors_concatenated.cpu().numpy())
    train_labels = train_ds_original.targets

    svm_classifier = SVC(kernel='linear', C=8)
    svm_classifier.fit(train_features, train_labels)

    # Evaluate the SVM classifier on the validation dataset
    test_svm_vectors = []

    for data, _ in val_ds_original:
        feature_map = extract_feature_maps(encoder, data,device=device)
        test_svm_vectors.append(feature_map)

    test_svm_vectors_concatenated = torch.cat(test_svm_vectors, dim=0)
    test_svm_vectors_concatenated = test_svm_vectors_concatenated.to(device)

    test_features = scaler.transform(test_svm_vectors_concatenated.cpu().numpy())
    test_labels = val_ds_original.targets

    predictions = svm_classifier.predict(test_features)
    # decision_scores = svm_classifier.decision_function(test_features)
    # test_labels_binarized = label_binarize(test_labels, classes=svm_classifier.classes_)

    accuracy = accuracy_score(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)
    macro_precision = precision_score(test_labels, predictions, average='macro')
    macro_recall = recall_score(test_labels, predictions, average='macro')
    macro_f1 = f1_score(test_labels, predictions, average='macro')

    return accuracy,conf_matrix,macro_precision,macro_recall,macro_f1,svm_classifier,scaler


##If you want to use the old trained model, use net functions below instead of Class
def Resnet101(device='cuda'):
    net = models.resnet101(pretrained=True).to(device)
    for param in net.parameters():
        param.requires_grad = False

    custom_fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, 1024),  # Update input size based on the feature map size of ResNet101
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2)

    )
    resnet101 = nn.Sequential(*list(net.children())[:-1], custom_fc).to(device)
    return resnet101

def Resnet152(device='cuda'):
    net = models.resnet152(pretrained=True).to(device)
    for param in net.parameters():
        param.requires_grad = False

    custom_fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, 1024),  # Update input size based on the feature map size of ResNet101
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2)

    )
    resnet152 = nn.Sequential(*list(net.children())[:-1], custom_fc).to(device)
    return resnet152
