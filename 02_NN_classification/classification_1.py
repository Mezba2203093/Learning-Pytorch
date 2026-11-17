# %%
# dataset
import importlib
from helper_functions import plot_predictions, plot_decision_boundary
import requests
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import sklearn
from sklearn.datasets import make_circles
import pandas as pd
# make 1000 samples
n_samples = 1000

# create circles
x, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
len(x), len(y)
# %%
print(f"5samples of X:  {x[:5]}")
print(f"5samples of Y:  {y[:5]}")
# %%
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title("make_circles dataset (Red = Class 0, Blue = Class 1)")
plt.xlabel("Feature 1 (x-coordinate)")
plt.ylabel("Feature 2 (y-coordinate)")
plt.show()
# %%
circles = pd.DataFrame({"X1": x[:, 0],
                        "X2": x[:, 1],
                        "label": y})
circles.head(10)

# %%
# 1. input and output shapes
x.shape, y.shape
# %%
# 1.2 turn data into tensor
X = torch.from_numpy(x).to(dtype=torch.float32)
Y = torch.from_numpy(y).to(dtype=torch.float32).unsqueeze(dim=1)

X[:5], Y[:5]

# %%
# split the data

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,  # 0.3 30 percent of data will be used for testing
                                                    random_state=42)
len(X_train), len(Y_train), len(X_test), len(Y_test)


# %%
# building model
device = "cuda" if torch.cuda.is_available() else "cpu"


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=5)
        self.layer_2 = nn.Linear(in_features=5,
                                 out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.relu(self.layer_1(x)))


X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(
    device), Y_train.to(device), Y_test.to(device)

model_1 = CircleModelV0().to(device)
model_1.state_dict()

# %%
# building model
device = "cuda" if torch.cuda.is_available() else "cpu"
# dont run if the prev cell is on


class CircleModelV0_diff(nn.Module):
    def __init__(self):
        super().__init__()
        self.two_linear_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.two_linear_layers(x)


X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(
    device), Y_train.to(device), Y_test.to(device)

model_0 = CircleModelV0_diff().to(device)
model_0.state_dict()
# %%
X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(
    device), Y_train.to(device), Y_test.to(device)
with torch.inference_mode():
    untrainedy_y_pred = model_0(X_test)
print(
    f"Length of predictions:{len(untrainedy_y_pred)}, shape: {untrainedy_y_pred.shape}")
print(f"Length of samples:{len(X_test)}, shape: {X_test.shape}")
print(f"first 10 preds: {torch.round(abs(untrainedy_y_pred[:10]))}")
print(f"first 10 lables: {torch.round(Y_test[:10])}")
# %%
# setting up loss function and optimizer

loss_fn = nn.BCEWithLogitsLoss()  # sigmoid activation function builtin

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc


# %%
# used for testing what is logit what is prediction probability and what is predictions
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test)[:5]

# Use the sigmoid activation function on our model logits to turn them into prediction probability
y_pred_probs = torch.sigmoid(y_logits)
y_pred_probs

# Find the predicted labels
y_preds = torch.round(y_pred_probs)
y_preds

# %%
# In Full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test)[:5]))

# check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# get rid of extra dimension
y_preds.squeeze(), Y_test[:5]

# %%
# training and testing loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set the number of epochs
epochs = 1000

# put the data to the target device
X_train, Y_train = X_train.to(device), Y_train.to(device).squeeze()
X_test, y_test = X_test.to(device), Y_test.to(device).squeeze()

# Buillding training and evaluation loop
for epoch in range(epochs):
    # Training
    model_0.train()

    # 1. Forward Pass
    y_logits = model_0(X_train).squeeze()
    # turn logits -> pred probs _. pred labels
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), Y_train) # nn.BCELoss expects prediction probabilities

    loss = loss_fn(y_logits,  # nn.BCEWithLogitLoss expects raw logits as input
                   Y_train)
    acc = accuracy_fn(y_true=Y_train,
                      y_pred=y_pred)

    # 3. optimizer
    optimizer.zero_grad()

    # 4. backpropagation
    loss.backward()

    # 5> optimizer step
    optimizer.step()

    # testing
    model_0.eval()

    with torch.inference_mode():
        # 1. Forward Pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate test loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # print out whats happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | train_loss: {loss:.5f} | train_acc: {acc:.2f} | Test loss: {test_loss:.5f} | Test Acc: {test_acc: .2f}")

# %%

# download helper functions from Learn PyTorch repo (if its not already downloaded)
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists")
else:
    print("Downloading helper_functions.py")
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)


# %%
# plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, Y_train.squeeze())
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, Y_test.squeeze())

# %%
# create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# model save path
MODEL_NAME = "Circle_classification.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

# save model state dict
print(f"saving model to : {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)
# %%

# using circle model to fit a stright line
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias
X[:10], Y[:10], len(X), len(Y)

# %%
# create a train/test split (for training and testing)
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]
# Duplicate the column so shape becomes (40, 2) <---
X_train = torch.cat([X_train, X_train], dim=1)
X_test = torch.cat([X_test, X_test], dim=1)
len(X_train), len(Y_train), len(X_test), len(Y_test)
# %%
# data visualize


def plot_prediction(train_data=X_train[:, 0],
                    train_label=Y_train,
                    test_data=X_test[:, 0],
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
torch.manual_seed(10)
model_2 = CircleModelV0_diff()
list(model_2.parameters())

# %%
loss_fn = nn.L1Loss()

loss_fn

# optimizer
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01)

# %%
epochs = 100
# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []


# 0. loop through the data
for epoch in range(epochs):
    # set the model to training mode
    model_2.train()

    # 1. forward pass
    y_pred = model_2(X_train)

    # 2 calculate the loss
    loss = loss_fn(y_pred, Y_train)
    # print(f"loss = {loss}")
    # 3 optimizer zero grad
    optimizer.zero_grad()

    # 4 perfrom backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5 step the optimizer
    optimizer.step()

    # testing
    model_2.eval()  # turns off different settings in the model not needed for evaluating /testing (dropout/batch.norm)
    with torch.inference_mode():  # turns off gradient tracking & a couple of more things
        # 1. still do forward pass
        test_preds = model_2(X_test)

        # calculate the loss
        test_loss = loss_fn(test_preds, Y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f"epoch : {epoch} | loss : {loss} | test loss: {test_loss}")
        # print out model state_dict
        print(model_2.state_dict())

# %%
model_2.state_dict()
# %%
plot_prediction(predictions=test_preds.squeeze())
# %%
plt.plot(epoch_count, train_loss_values, label="train loss")
plt.plot(epoch_count, test_loss_values, label="test loss")
plt.title("Training and test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# %%
importlib.reload(torch)

torch.manual_seed(42)   # works again
# %%
