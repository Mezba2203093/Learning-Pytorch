# %%
from helper_functions import plot_predictions, plot_decision_boundary
import requests
import sklearn
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
Num_Classes = 4
Num_Features = 2
Rand_seed = 42

# creating multiclass data
X_blob, Y_blob = make_blobs(n_samples=1000,
                            n_features=Num_Features,
                            centers=Num_Classes,
                            cluster_std=1.5,
                            random_state=Rand_seed)

# turn data into tensors
X_blob = torch.from_numpy(X_blob).to(dtype=torch.float)
Y_blob = torch.from_numpy(Y_blob).to(dtype=torch.long)

# split into train and test
X_blob_train, X_blob_test, Y_blob_train, Y_blob_test = train_test_split(X_blob,
                                                                        Y_blob,
                                                                        test_size=0.3,
                                                                        random_state=Rand_seed)
# plotting
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0].cpu(), X_blob[:, 1].cpu(),
            c=Y_blob.cpu(), cmap=plt.cm.RdYlBu)

# %%
# multiclass classification model
X_blob_train, X_blob_test = X_blob_train.to(device), X_blob_test.to(device)
Y_blob_train, Y_blob_test = Y_blob_train.to(device), Y_blob_test.to(device)


class BlobModel_v0(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)


# %%
model_0 = BlobModel_v0(input_features=2,
                       output_features=4,
                       hidden_units=128).to(device)
model_0
# %%
# loss func and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc


# %%
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_blob_test)
    y_pred_prob = torch.softmax(y_logits, dim=1)
    y_preds = torch.argmax(y_pred_prob, dim=1)
y_logits[:5], y_pred_prob[:5], y_preds[:5], Y_blob_test[:5]
# %%

epochs = 3000

for epoch in range(epochs):
    model_0.train()

    # forward pass
    y_logits = model_0(X_blob_train)
    y_pred_prob = torch.softmax(y_logits, dim=1)
    train_preds = torch.argmax(y_pred_prob, dim=1)

    # calculate the loss
    loss = loss_fn(y_logits,
                   Y_blob_train)

    train_acc = accuracy_fn(Y_blob_train, train_preds)

    # optimizer zero grad
    optimizer.zero_grad()

    # calculate the backward loss
    loss.backward()

    # step
    optimizer.step()

    # testing loop

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_blob_test)
        test_pred_prob = torch.softmax(test_logits, dim=1)
        test_preds = torch.argmax(test_pred_prob, dim=1)
        test_loss = loss_fn(test_logits,
                            Y_blob_test)

    test_acc = accuracy_fn(Y_blob_test, test_preds)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | "
              f"train loss {loss.item():.5f} acc {train_acc:.3f} | "
              f"test loss {test_loss.item():.5f} acc {test_acc:.3f}")
# %%
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
plot_decision_boundary(model_0, X_blob_train, Y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_blob_test, Y_blob_test)
# %%
