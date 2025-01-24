import numpy as np
import random

def train_val_test_split(X_data, y_data, model_num):
    X = np.copy(np.array(X_data))
    y = np.copy(np.array(y_data))
    X_test = X[model_num]
    y_test = y[model_num]

    X = np.delete(X, model_num, axis=0)
    y = np.delete(y, model_num, axis=0)

    temp = list(zip(X, y))

    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)

    X_train = []
    y_train = []

    for i in range(0, 7):
        X_train.extend(res1[i])
        y_train.extend(res2[i])

    X_val = []
    y_val = []

    for i in range(7, 9):
        X_val.extend(res1[i])
        y_val.extend(res2[i])

    return np.array(X_test), np.array(y_test), np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

