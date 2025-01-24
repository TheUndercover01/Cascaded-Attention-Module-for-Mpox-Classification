import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from data_splitter import train_val_test_split
from train_test_functions import train_one_epoch, test
from utils import norm, replace_context_modules, evaluate, split_dataset, acc_eval, cal_total
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import timm
from loss import FocalCELoss, EarlyStopper
from model import CascadedAttentionModule, change_classifier



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_labels = {
    'monkey_pox': [1, 0, 0, 0],
    'normal': [0, 1, 0, 0],
    'chicken_pox': [0, 0, 1, 0],
    'acne': [0, 0, 0, 1],
}


def training(run, EPOCH=55, k=10):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create a directory for saving CSV files
    csv_dir = f'./results/run_{run}'
    os.makedirs(csv_dir, exist_ok=True)

    # Create a CSV file for this run
    csv_file_path = os.path.join(csv_dir, f'results_{timestamp}.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        header = ['Model', 'Fold', 'Accuracy', 'Precision', 'Recall', 'F1'] + [f'{class_name} Accuracy' for class_name in class_labels.keys()]
        csvwriter.writerow(header)

    X_data = torch.Tensor(np.load('/content/drive/MyDrive/X_train_final_multi_10_folds_40_each_equal.npy', allow_pickle=True))
    y_data = torch.Tensor(np.load('/content/drive/MyDrive/y_train_final_multi_10_folds_40_each__equal.npy', allow_pickle=True))

    print(X_data.shape)

    # Define the list of models to train
    models_to_train = [
        ('efficientvit_l1.r224_in1k', None),
        #('deit3_small_patch16_224', None),
        #('resnet50', None),
        #('vit_base_patch16_224', None),
        #('swin_base_patch4_window7_224', None)
    ]

    for model_name, custom_head in models_to_train:
        accuracy_dict_train = {}
        accuracy_dict_val = {}
        loss_dict_train = {}
        loss_dict_val = {}
        accu = []
        print(f"Training {model_name}")

        accuracy_avg = 0
        precision_avg = 0
        recall_avg = 0
        f1_avg = 0

        class_total_test_acc = {class_name: 0 for class_name in class_labels.keys()}

        for model_num in range(k):
            print(f"####################################")
            print(f"K IS {model_num}")

            print(f"X_data shape: {X_data.shape}")
            print(f"y_data type: {type(y_data)}")

            epoch_number = 0
            best_vloss = 1_000_000.

            model = timm.create_model('efficientvit_l1.r224_in1k', pretrained=True)
            replace_context_modules(model, CascadedAttentionModule)
            change_classifier(model,'efficientvit_l1.r224_in1k', dropout=0.111975, neurons1=3072, neurons2=3200, neurons3=512, neurons4=512, n_layers=2)

            print(model)
            model.to(device)
            print(12)

            X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_data, y_data, test_fold=model_num, val_folds=2)
            X_train, X_val, X_test = norm(X_test, X_train, X_val)

            training_loader = DataLoader(list(zip(torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device))), batch_size=16, shuffle=True)
            validation_loader = DataLoader(list(zip(torch.Tensor(X_val).to(device), torch.Tensor(y_val).to(device))), batch_size=16, shuffle=True)
            test_loader = DataLoader(list(zip(torch.Tensor(X_test).to(device), torch.Tensor(y_test).to(device))), batch_size=8, shuffle=False)

            alpha = alpha = torch.Tensor([0.1406, 0.1007, 0.1394, 0.0872]).to(device)

            loss_fn = torch.nn.BCELoss()
            loss_fn1 = FocalCELoss(alpha = alpha)
            #loss_fn1 = torch.nn.CrossEntropyLoss()
            w = 1
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00009613447274, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            early_stopper = EarlyStopper(patience=9, min_delta=0)

            print(f'Model {model_num}:')

            for epoch in range(EPOCH):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                print(f'EPOCH {epoch_number + 1}:')

                model.train(True)
                avg_loss, train_accuracy, right, *class_accuracies_train = train_one_epoch(model, epoch_number, model_num, training_loader, loss_fn, loss_fn1, w, optimizer, loss_dict_train)

                print(f"Average loss: {avg_loss}")

                running_vloss = 0.0
                model.eval()
                vright_total = 0
                total_val = 0

                class_totals_val = {class_name: 0 for class_name in class_labels.keys()}
                class_rights_val = {class_name: 0 for class_name in class_labels.keys()}

                with torch.no_grad():
                    for i, vdata in enumerate(validation_loader):
                        vinputs, vlabels = vdata
                        total_val += vinputs.shape[0]

                        voutputs = model(vinputs)
                        vloss = (1-w) * loss_fn(voutputs, vlabels) + w * loss_fn1(voutputs, vlabels)

                        class_totals_batch = cal_total(vlabels)
                        vright, *class_rights_batch = acc_eval(voutputs, vlabels, classwise=True)

                        vright_total += vright

                        for j, class_name in enumerate(class_labels.keys()):
                            class_totals_val[class_name] += class_totals_batch[j]
                            class_rights_val[class_name] += class_rights_batch[j]

                        running_vloss += vloss

                avg_vloss = running_vloss / (i + 1)
                print(f"Validation loss: {avg_vloss}")

                class_accuracies_val = {}
                for class_name in class_labels.keys():
                    if class_totals_val[class_name] == 0:
                        class_accuracies_val[class_name] = None
                    else:
                        class_accuracies_val[class_name] = class_rights_val[class_name] / class_totals_val[class_name]

                scheduler.step()

                val_acc = vright_total / total_val

                print(f"Validation accuracy: {val_acc}")
                print(f'LOSS train {avg_loss} valid {avg_vloss}')
                print(f'Right train {right} valid {vright_total}')
                print(f'Accuracy train {train_accuracy} valid {val_acc}')

                print('---------->classwise<-----------')
                for class_name in class_labels.keys():
                    print(f'Accuracy {class_name} train {class_accuracies_train[list(class_labels.keys()).index(class_name)]} valid {class_accuracies_val[class_name]}')

                epoch_number += 1
                early = early_stopper.early_stop(val_acc)
                print("Current early_stop count", early[1])
                if early[0]:
                    print('Early stopping triggered')
                    break

            with torch.no_grad():
                acc, precision, recall, f1, *class_accuracies_test = test(model, test_loader, model_num)
                print('##########################################################')
                print(f'Accuracy test {acc}')
                print(f'Precision test {precision}')
                print(f'Recall test {recall}')
                print(f'F1 test {f1}')

                print('---------->classwise test<-----------')
                for i, class_name in enumerate(class_labels.keys()):
                    print(f'Accuracy {class_name} test {class_accuracies_test[i]}')

                # Save results to CSV
                with open(csv_file_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    row = [model_name, model_num, acc, precision, recall, f1] + class_accuracies_test
                    csvwriter.writerow(row)

            accuracy_avg += acc
            precision_avg += precision
            recall_avg += recall
            f1_avg += f1

            for i, class_name in enumerate(class_labels.keys()):
                class_total_test_acc[class_name] += class_accuracies_test[i]

        accuracy_avg /= k
        precision_avg /= k
        recall_avg /= k
        f1_avg /= k

        for class_name in class_labels.keys():
            class_total_test_acc[class_name] /= k

        print(f'Average results for {model_name}:')
        print(f'Average Accuracy test {accuracy_avg}')
        print(f'Average precision test {precision_avg}')
        print(f'Average recall test {recall_avg}')
        print(f'Average f1 test {f1_avg}')

        print('---------->classwise test Average<-----------')
        for class_name, avg_acc in class_total_test_acc.items():
            print(f'Average {class_name} Accuracy test {avg_acc}')

        # Save average results to CSV
        with open(csv_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            avg_row = [model_name, 'Average', accuracy_avg, precision_avg, recall_avg, f1_avg] + list(class_total_test_acc.values())
            csvwriter.writerow(avg_row)

if __name__ == '__main__':
    training(0)