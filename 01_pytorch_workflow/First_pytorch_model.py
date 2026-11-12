# %%
import torch
import matplotlib.pyplot as plt
import numpy
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
# create linear regression


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
Y_test
# %%
plot_prediction(predictions=y_preds)


# %%
y_pred = model_0(X_test)
y_pred

# %%
# training model
model_0.state_dict()
# loss function
loss_fn = nn.L1Loss()

loss_fn

# optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
# %%
torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, Y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    # Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
        # 1. Forward pass on test data
        test_pred = model_0(X_test)

        # 2. Caculate loss on test data
        # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
        test_loss = loss_fn(test_pred, Y_test.type(torch.float))

        # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(
                f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# %%
# loss curves
plt.plot(epoch_count, train_loss_values, label="train loss")
plt.plot(epoch_count, test_loss_values, label="test loss")
plt.title("Training and test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# %%
# predictions after training
model_0.eval()
with torch.inference_mode():
    y_preds_new = model_0(X_test)
plot_prediction(predictions=y_preds_new)

# %%
model_0.state_dict()  # should be close to weights 0.7, bias 0.3
# %%
