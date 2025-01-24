import numpy as np
import torch
from tqdm import tqdm
import torch
from data_splitter import EarlyStopper, train_val_test
#from train_test_functions import train_one_epoch, test
from utils import norm, replace_context_modules, evaluate, split_dataset, acc_eval, cal_total

def test(model, training_loader, model_num):
    right_total, total = 0, 0
    out, lab = [], []
    monkey_pox_total, normal_total, chicken_pox_total, acne_total = 0, 0, 0, 0
    monkey_pox_right, normal_right, chicken_pox_right, acne_right = 0, 0, 0, 0

    for i, data in enumerate(tqdm(training_loader)):
        inputs, labels = data
        total += inputs.shape[0]
        outputs = model(inputs)

        mpt, nt, cpt, at = cal_total(outputs, labels)
        right, mpr, nr, cpr, ar = acc_eval(outputs, labels, classwise=True)

        right_total += right
        monkey_pox_total += mpt
        normal_total += nt
        chicken_pox_total += cpt
        acne_total += at
        monkey_pox_right += mpr
        normal_right += nr
        chicken_pox_right += cpr
        acne_right += ar

        outputs = np.array(outputs.detach().cpu(), dtype='object')
        labels = np.array(labels.detach().cpu(), dtype='object')
        out.extend(np.argmax(outputs, axis=1))
        lab.extend(np.argmax(labels, axis=1))

    precision, recall, f1 = evaluate(out, lab)
    accuracy = right_total / total
    monkey_pox_acc = monkey_pox_right / monkey_pox_total
    normal_acc = normal_right / normal_total
    chicken_pox_acc = chicken_pox_right / chicken_pox_total
    acne_acc = acne_right / acne_total

    print("Accuracy", accuracy)
    print("Total Right", right_total)

    return accuracy, precision, recall, f1, monkey_pox_acc, normal_acc, chicken_pox_acc, acne_acc

def train_one_epoch(model, epoch_index, model_num, training_loader, loss_fn, loss_fn1, w, optimizer, loss_dict_train):
    running_loss, last_loss = 0., 0.
    right_total, total = 0, 0
    monkey_pox_total, normal_total, chicken_pox_total, acne_total = 0, 0, 0, 0
    monkey_pox_right, normal_right, chicken_pox_right, acne_right = 0, 0, 0, 0

    for i, data in enumerate(tqdm(training_loader)):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        total += inputs.shape[0]

        loss = (1 - w) * loss_fn(outputs, labels) + w * loss_fn1(outputs, labels)
        loss.backward()
        optimizer.step()

        mpt, nt, cpt, at = cal_total(outputs, labels)
        right, mpr, nr, cpr, ar = acc_eval(outputs, labels, classwise=True)

        right_total += right
        monkey_pox_total += mpt
        normal_total += nt
        chicken_pox_total += cpt
        acne_total += at
        monkey_pox_right += mpr
        normal_right += nr
        chicken_pox_right += cpr
        acne_right += ar

        running_loss += loss.item()
        loss_dict_train[model_num].append(loss)
        if i % 10 == 9:
            last_loss = running_loss / 10
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    monkey_pox_acc_train = monkey_pox_right / monkey_pox_total if monkey_pox_total > 0 else None
    normal_acc_train = normal_right / normal_total if normal_total > 0 else None
    chicken_pox_acc_train = chicken_pox_right / chicken_pox_total if chicken_pox_total > 0 else None
    acne_acc_train = acne_right / acne_total if acne_total > 0 else None

    return last_loss, right_total / total, right_total, monkey_pox_acc_train, normal_acc_train, chicken_pox_acc_train, acne_acc_train