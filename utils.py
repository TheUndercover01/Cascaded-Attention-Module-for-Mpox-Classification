import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, random_split, DataLoader

class_labels = {
    'monkey_pox': [1, 0, 0, 0],
    'normal': [0, 1, 0, 0],
    'chicken_pox': [0, 0, 1, 0],
    'acne': [0, 0, 0, 1],
}



def replace_context_modules(model, MyModule):
    # Access the fourth stage (index 3)
    stage = model.stages[3]

    # Replace context_module in blocks 1 through 5
    for i in range(1, 7):
        block = stage.blocks[i]

        # Get the input and output channels from the original context_module
        in_channels = block.context_module.main.qkv.conv.in_channels
        out_channels = block.context_module.main.proj.conv.out_channels

        # Replace the context_module with your custom module
        block.context_module = nn.Sequential(
            MyModule(in_channels, nn.ReLU)
        )




def evaluate(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.2):
    assert len(X) == len(y), "The number of samples in X and y must be the same."
    dataset = TensorDataset(X, y)
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size)
    dataset = torch.utils.data.Subset(dataset, indices)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    X_train = torch.stack([dataset[i][0] for i in train_dataset.indices])
    y_train = torch.stack([dataset[i][1] for i in train_dataset.indices])
    X_val = torch.stack([dataset[i][0] for i in val_dataset.indices])
    y_val = torch.stack([dataset[i][1] for i in val_dataset.indices])
    X_test = torch.stack([dataset[i][0] for i in test_dataset.indices])
    y_test = torch.stack([dataset[i][1] for i in test_dataset.indices])
    return X_train, y_train, X_val, y_val, X_test, y_test

def acc_eval(outputs, labels, classwise=False):
    if not classwise:
        right = 0
        for j in range(outputs.shape[0]):
            max_value = torch.max(outputs[j])
            outputs[j] = (outputs[j] == max_value).float()
            if list(outputs[j]) == list(labels[j]):
                right += 1
        return right
    else:
        right, monkey_pox_right, normal_right, chicken_pox_right, acne_right = 0, 0, 0, 0, 0
        for j in range(outputs.shape[0]):
            max_value = torch.max(outputs[j])
            outputs[j] = (outputs[j] == max_value).float()
            if list(outputs[j]) == list(labels[j]):
                right += 1
                if list(labels[j].detach().cpu()) == class_labels['monkey_pox']:
                    monkey_pox_right += 1
                elif list(labels[j].detach().cpu()) == class_labels['normal']:
                    normal_right += 1
                elif list(labels[j].detach().cpu()) == class_labels['chicken_pox']:
                    chicken_pox_right += 1
                elif list(labels[j].detach().cpu()) == class_labels['acne']:
                    acne_right += 1
        return right, monkey_pox_right, normal_right, chicken_pox_right, acne_right

def cal_total(outputs, labels):
    monkey_pox_total = sum(1 for j in range(outputs.shape[0]) if list(labels[j].detach().cpu()) == class_labels['monkey_pox'])
    normal_total = sum(1 for j in range(outputs.shape[0]) if list(labels[j].detach().cpu()) == class_labels['normal'])
    chicken_pox_total = sum(1 for j in range(outputs.shape[0]) if list(labels[j].detach().cpu()) == class_labels['chicken_pox'])
    acne_total = sum(1 for j in range(outputs.shape[0]) if list(labels[j].detach().cpu()) == class_labels['acne'])
    return monkey_pox_total, normal_total, chicken_pox_total, acne_total

def norm(X_test, X_train, X_val):
    meanx = X_train.mean()
    stdx = X_train.std()
    X_train = (X_train - meanx) / stdx
    X_valid = (X_val - meanx) / stdx
    X_test = (X_test - meanx) / stdx
    return X_train, X_valid, X_test