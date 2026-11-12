
# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

# %%
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias
X[:10], Y[:10], len(X), len(Y)

# %%
# create a train/test split
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]
len(X_train), len(Y_train), len(X_test), len(Y_test)

# %%
# data visualize


def plot_prediction(train_data=X_train,
                    train_label=Y_train,
                    test_data=X_test,
                    test_label=Y_test,
                    predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c="b", s=4, label="training data")
    plt.scatter(test_data, test_label, c="g", s=4, label="test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

    plt.legend(prop={"size": 14})


plot_prediction()
# %%


# %%
