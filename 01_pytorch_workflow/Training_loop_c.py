# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

# %%
# Ground truth — the relationship our model must learn to reproduce.
# The model will start with random w and b and should converge to these values.
weight = 0.7
bias = 0.3

# Build a simple linear dataset: Y = 0.7 * X + 0.3
start = 0
end = 1
step = 0.02
# (50,) -> (50, 1): models expect 2D input
X = torch.arange(start, end, step).unsqueeze(dim=1)
# target values (no noise, so it's easy to fit)
Y = weight * X + bias
X[:10], Y[:10], len(X), len(Y)

# %%
# Train/test split — 80% of the data for training, 20% held out for testing.
# The test set is never seen during training, so it measures generalization.
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test,  Y_test = X[train_split:], Y[train_split:]
len(X_train), len(Y_train), len(X_test), len(Y_test)

# %%
# Visualize the data: blue = training points, green = test points.
# The model should learn a line through the blue points that also fits the green ones.


def plot_prediction(train_data=X_train,
                    train_label=Y_train,
                    test_data=X_test,
                    test_label=Y_test,
                    predictions=None):
    """
    Plots training data, test data and (optionally) predictions on the test set.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c="b", s=4, label="training data")
    plt.scatter(test_data,  test_label,  c="g", s=4, label="test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

    plt.legend(prop={"size": 14})


plot_prediction()

# %%
# Model: y = w * x + b, where w and b are learned from data.
# Almost everything in PyTorch inherits from nn.Module, which handles
# parameter registration and the forward/backward plumbing.


class LinearRegressionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # nn.Parameter -> tensor auto-registered as trainable (requires_grad=True by default),
        # so model.parameters() will hand it to the optimizer.
        self.weights = nn.Parameter(torch.rand(
            1, dtype=torch.float))  # random starting w
        self.bias = nn.Parameter(torch.rand(
            1, dtype=torch.float))  # random starting b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The linear regression formula. This is the "computation" the model performs.
        return self.weights * x + self.bias


# %%
# Manual seed makes the random init reproducible (same w and b every run).
torch.manual_seed(10)
model_0 = LinearRegressionModel()
list(model_0.parameters())

# %%
# state_dict shows the current (random) values of w and b — far from 0.7 and 0.3.
model_0.state_dict()

# %%
# Make predictions BEFORE training, to see the starting point.
# inference_mode() disables autograd: we're not training, so no gradients needed.
with torch.inference_mode():
    y_preds = model_0(X_test)

y_preds

# %%
# Plot the untrained model's predictions — they should look far off the data.
plot_prediction(predictions=y_preds)

# %%
# Loss function: L1Loss = Mean Absolute Error.
# MAE is chosen for its simplicity and robustness to outliers;
# MSE (nn.MSELoss) would also work for this clean data.
loss_fn = nn.L1Loss()

# Optimizer: Stochastic Gradient Descent.
# It updates w and b along the negative gradient of the loss, scaled by the learning rate:
#     param = param - lr * grad(param)
# lr=0.01 is small enough to converge smoothly on this simple data.
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# %%
# Training loop. An "epoch" is one full pass over the training data.
# 300 epochs is enough for SGD at lr=0.01 to converge on this simple linear problem.

epochs = 300

for epoch in range(epochs):
    # 0. Set model to training mode (matters for dropout/batchnorm; harmless here).
    model_0.train()

    # 1. Forward pass: compute predictions on the training inputs.
    y_pred = model_0(X_train)

    # 2. Compute the loss between predictions and true targets.
    loss = loss_fn(y_pred, Y_train)

    # Print occasionally so the output shows learning, not 300 lines of noise.
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f} "
              f"| w {model_0.weights.item():.3f} | b {model_0.bias.item():.3f}")

    # 3. Zero the gradients — PyTorch accumulates them by default,
    #    so they must be cleared before each backward pass.
    optimizer.zero_grad()

    # 4. Backpropagation: compute dLoss/dw and dLoss/db via autograd.
    loss.backward()

    # 5. Update w and b using the optimizer's rule.
    optimizer.step()

# %%
# After training, w and b should be close to the true values (0.7 and 0.3).
model_0.state_dict()

# %%
# Switch to eval mode and disable autograd for inference.
model_0.eval()
with torch.inference_mode():
    y_preds_new = model_0(X_test)

# %%
# Compare final predictions (red) against the true test points (green).
# If training worked, they should lie almost on top of each other.
plot_prediction(predictions=y_preds_new)

# %%
