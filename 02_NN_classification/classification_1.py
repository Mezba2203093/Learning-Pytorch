# %%
# dataset
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x))


model_0 = CircleModelV0().to(device)
model_0

# %%


class CircleModelV0_diff(nn.Module):
    def __init__(self):
        super().__init__()
        self.two_linear_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.two_linear_layers(x)


model_1 = CircleModelV0().to(device)
model_1
# %%
X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(
    device), Y_train.to(device), Y_test.to(device)
with torch.inference_mode():
    y_pred = model_0(X_train)
y_pred
# %%
