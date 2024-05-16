from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
import requests
import torch

import torch.nn as nn

if Path("../helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("../helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

X, y = make_moons(1000, random_state=42)

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

"""
plt.figure(figsize=(7,7))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = (X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device))


class MakeMoonsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(out_features=10, in_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


model_0 = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10,1)
).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(model_0.parameters(), lr=0.1)
acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

epochs = 10000

for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass
    y_logits = model_0(X_train).squeeze() # model outputs raw logits
    y_pred = torch.round(torch.sigmoid(y_logits)) # go from logits -> prediction probabilities -> prediction labels


    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train.int())

    # 3. Optimizer zero grad
    optim.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optim.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_0(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits))
      # 2. Calculate test loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = acc_fn(test_pred, y_test.int())

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)


plt.figure(figsize=(7,7))
plot_decision_boundary(model_0, X_test, y_test)
plt.show()