# %%
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

device = "cuda " if torch.cuda.is_available() else "cpu"
# %%
# ground truth - that is to be matched by predictions
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

# model to make predictions


# <- almost everything in putorch inherits from nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.rand(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1,
                                            requires_grad=True,
                                            dtype=torch.float))
        # forward method to define the compution in the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- x is the input data
        return self.weights * x + self.bias  # linear regression formula


# %%
# using manual seed to reduce the randomness
torch.manual_seed(10)
model_0 = LinearRegressionModel()
list(model_0.parameters())

# %%
model_0.state_dict()

# %%
# making predictions
with torch.inference_mode():
    y_preds = model_0(X_test)

y_preds

# %%
plot_prediction(predictions=y_preds)

# %%
loss_fn = nn.L1Loss()

loss_fn

# optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# %%
# an epoch is one loop through thhe data

epochs = 100
# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []


# 0. loop through the data
for epoch in range(epochs):
    # set the model to training mode
    model_0.train()

    # 1. forward pass
    y_pred = model_0(X_train)

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
    model_0.eval()  # turns off different settings in the model not needed for evaluating /testing (dropout/batch.norm)
    with torch.inference_mode():  # turns off gradient tracking & a couple of more things
        # 1. still do forward pass
        test_preds = model_0(X_test)

        # calculate the loss
        test_loss = loss_fn(test_preds, Y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f"epoch : {epoch} | loss : {loss} | test loss: {test_loss}")
        # print out model state_dict
        print(model_0.state_dict())

# %%
model_0.state_dict()
# %%
plot_prediction(predictions=test_preds)
# %%
plt.plot(epoch_count, train_loss_values, label="train loss")
plt.plot(epoch_count, test_loss_values, label="test loss")
plt.title("Training and test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# %%
# saving model

# create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

# save model state dict
print(f"saving model to : {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)


# %%
# loading
loaded_model_0 = LinearRegressionModel()

# load the sate dict of model_0

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print("Loaded model weights:", loaded_model_0.state_dict())

# %%

# making prediction with loaded_model_0
X_1 = torch.arange(0, 3, 0.02).unsqueeze(dim=1)

# Generate the ground truth labels for X_1 (length 150)
Y_1 = 0.7 * X_1 + 0.3


loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_1)

loaded_model_preds, X_1.device

# %%
# compare loaded model preds


# FIX: Pass Y_1 as test_label so the lengths match (150 and 150)
plot_prediction(test_data=X_1, test_label=Y_1, predictions=loaded_model_preds)


# %%
