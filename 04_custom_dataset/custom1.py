# %%
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchinfo import summary
from typing import Tuple, Dict, List
from torchvision import transforms
from torch.utils.data import Dataset
import os
from torchvision import datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from PIL import Image
import random
import requests
import zipfile
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
# setup data path
data_path = Path("../04_custom_dataset/data/")
image_path = data_path / "Pizza_Steak_Sushi"


# if the img folder doesnt exist we want to download and unzip

if image_path.is_dir():
    print(f"{image_path} directory already exists")
else:
    print(f"{image_path} doesnt exists, creating")
    image_path.mkdir(parents=True, exist_ok=True)

# downloading data
with open(data_path / "Pizza_Steak_Sushi.zip", "wb") as f:
    request = requests.get(
        "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print(f"Downloading pizza steak sushi data")
    f.write(request.content)

# unzip pizza steak sushi data
with zipfile.ZipFile(data_path/"Pizza_Steak_Sushi.zip", "r") as zip_ref:
    print(f"unzipping pizza steak and sushi data")
    zip_ref.extractall(image_path)
# %%


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"there are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")


walk_through_dir(image_path)
# %%
# visualizing data


# %%


# Write transform for image
data_transform_1 = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    # p = probability of flip, 0.5 = 50% chance
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn the image into a torch.Tensor
    # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    transforms.ToTensor()
])

# %%

# %%
# Use ImageFolder to create dataset(s)

train_dir = data_path/"Pizza_Steak_Sushi/train"
test_dir = data_path/"Pizza_Steak_Sushi/test"
train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
                                  # transforms to perform on data (images)
                                  transform=data_transform_1,
                                  # transforms to perform on labels (if necessary)
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform_1)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

# %%


# %%
# Turn train and test Datasets into DataLoaders
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=25,  # how many samples per batch?
                              # how many subprocesses to use for data loading? (higher = more)
                              num_workers=0,
                              shuffle=True)  # shuffle the data?

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=25,
                             num_workers=0,
                             shuffle=False)  # don't usually need to shuffle testing data

train_dataloader, test_dataloader

# %%


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted([entry.name for entry in list(os.scandir(directory))])
    if not classes:
        raise FileNotFoundError(f"couldnt find any classes in that directory")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx


find_classes(train_dir)
# %%


class ImageCustom(Dataset):
    def __init__(self,
                 targ_dir: str,
                 transfrom=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transfrom = transfrom
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self,
                   index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transfrom:
            return self.transfrom(img), class_idx
        else:
            return img, class_idx


# %%
# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# %%
train_data_custom = ImageCustom(targ_dir=train_dir,
                                transfrom=train_transforms)
test_data_custom = ImageCustom(targ_dir=test_dir,
                               transfrom=test_transforms)
train_data_custom, test_data_custom
# %%
Batch_Size = 25
Num_workers = os.cpu_count()
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=Batch_Size,
                                     num_workers=0,
                                     shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=Batch_Size,
                                    num_workers=0,
                                    shuffle=False)
# %%
# Augment train data
train_transforms_1 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms_1 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# %%
train_data_custom_1 = ImageCustom(targ_dir=train_dir,
                                  transfrom=train_transforms_1)
test_data_custom_1 = ImageCustom(targ_dir=test_dir,
                                 transfrom=test_transforms_1)
train_data_custom, test_data_custom
# %%
Batch_Size = 25
Num_workers = 0
train_dataloader_custom_1 = DataLoader(dataset=train_data_custom,
                                       batch_size=Batch_Size,
                                       num_workers=Num_workers,
                                       shuffle=True)

test_dataloader_custom_1 = DataLoader(dataset=test_data_custom,
                                      batch_size=Batch_Size,
                                      num_workers=Num_workers,
                                      shuffle=False)

# %%
len(train_dataloader), len(train_dataloader_custom), len(
    train_dataloader_custom_1)
# %%


class FoodModel_V1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_unit: int,
                 output_shape: int):
        super().__init__()
        self.conv_b1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_unit*16*16,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_b2(self.conv_b1(x)))


# %%
torch.manual_seed(42)
model_0 = FoodModel_V1(
    input_shape=3,
    hidden_unit=32,
    output_shape=len(train_data.classes)
).to(device)

model_0
# %%
summary(model_0, input_size=[1, 3, 64, 64])
# %%


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
# %%


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() /
                         len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# %%

# 1. Take in various parameters required for training and test steps


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(
            train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(
            train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(
            test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(
            test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


# %%
# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 20

# Recreate an instance of TinyVGG
model_0 = FoodModel_V1(input_shape=3,  # number of color channels (3 for RGB)
                       hidden_unit=16,
                       output_shape=len(train_data.classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
start_time = timer()

# Train model_0
model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_custom_1,
                        test_dataloader=test_dataloader_custom_1,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
# %%
# Check the model_0_results keys
model_0_results.keys()
# %%


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


# %%
plot_loss_curves(model_0_results)
# %%
