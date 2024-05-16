from pathlib import Path

from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

import requests
import torch
import torchvision
import random

import torch.nn as nn


from helper_functions import accuracy_fn

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    # Note: you need the "raw" GitHub URL for this to work
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

device = "cuda" if torch.cuda.is_available() else "cpu"


class CNNModelRecreation(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=10 * 7 * 7, out_features=11)
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optim: torch.optim.Optimizer):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (img, label) in enumerate(dataloader):
            X, y = img.to(device), label.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)

            train_loss += loss
            train_acc += accuracy_fn(y_pred=y_pred.argmax(dim=1), y_true=y)

            optim.zero_grad()

            loss.backward()

            optim.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = sample.to(device)  # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(),
                                      dim=0)  # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for img, label in dataloader:
            X, y = img.to(device), label.to(device)

            y_pred = model(X)

            test_loss += loss_fn(y_pred, y)
            test_acc += accuracy_fn(y_pred=y_pred.argmax(dim=1), y_true=y)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    print(f"Train Loss: {test_loss} | Train Acc: {test_acc}")


def eval_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):

    eval_loss, eval_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_preds = model(X)

            eval_loss += loss_fn(y_preds, y)
            eval_acc += accuracy_fn(y_true=y, y_pred=y_preds.argmax(dim=1))

        eval_loss /= len(dataloader)
        eval_acc /= len(dataloader)

    print(f"Eval Loss: {eval_loss} | Eval Acc: {eval_acc}")


def plot_images(data_to_display, rows: int, columns: int):
    fig = plt.figure(figsize=(9, 9))

    for i in range(1, rows * columns + 1):
        img, label = data_to_display[i]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img.squeeze())
        plt.title(label)
        plt.axis(False)

    plt.show()


train_data = torchvision.datasets.MNIST(root="data",
                                        download=True,
                                        train=True,
                                        transform=ToTensor())

test_data = torchvision.datasets.MNIST(root="data",
                                       download=True,
                                       train=False,
                                       transform=ToTensor())

plot_images(train_data, 1, 5)

Batch_size = 32

train_dataloader = DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=Batch_size, shuffle=False)

#Training & Eval

model = CNNModelRecreation()

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.1)

epochs = 3

for epoch in tqdm(range(epochs)):
    train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optim=optim)
    test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)

eval_model(model, dataloader=test_dataloader, loss_fn=loss_fn)

#Plot predictions

class_names = train_data.classes

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_dataloader), k=9):
    test_samples.append(sample)
    test_labels.append(label)


pred_probs = make_predictions(model=model,
                             data=test_samples)

pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create a subplot
    plt.subplot(nrows, ncols, i + 1)

    # Plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")

    # Find the prediction label (in text form, e.g. "Sandal")
    pred_label = class_names[pred_classes[i]]

    # Get the truth label (in text form, e.g. "T-shirt")
    truth_label = class_names[test_labels[i]]

    # Create the title text of the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # Check for equality and change title colour accordingly
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")  # green text if correct
    else:
        plt.title(title_text, fontsize=10, c="r")  # red text if wrong
    plt.axis(False)