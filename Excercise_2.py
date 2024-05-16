from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(1,1, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


weight = 0.3
bias = 0.9

start = 0
end = 1
step = 0.001

data = torch.arange(0, 1, 0.001, device=device).unsqueeze(dim=1)
y = weight * data + bias

train_split = int(0.8 * len(data))

x_train, x_test = data[:train_split], data[train_split:]
y_train, y_test = y[:train_split], y[train_split:]


def plot_data(train_data=x_train, test_data=x_test, train_labels=y_train, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data.cpu(), train_labels.cpu(), c="b", s=4, label="Train Data")

    plt.scatter(test_data.cpu(), test_labels.cpu(), c="g", s=4, label="Test Data")

    if predictions is not None:
        plt.scatter(test_data.cpu(), predictions.cpu(), c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show()


model = LinearRegression()

loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(lr=0.01, params=model.parameters())

print(model.state_dict())

epochs = 300

for epoch in range(epochs):
    model.train()

    y_preds = model(x_train)
    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    with torch.inference_mode():
        test_preds = model(x_test)
        test_loss = loss_fn(test_preds, y_test)

    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


with torch.inference_mode():
    preds = model(x_test)

plot_data(predictions=preds.cpu())

model_name = "Linear_Regression_Model.pth"
save_path = Path("models")

save_path.mkdir(parents=True, exist_ok=True)
model_save_path = save_path / model_name

torch.save(model.state_dict(), model_save_path)

model_2 = LinearRegression()
model_2.load_state_dict(torch.load(model_save_path))

model_2.to(device)
model_2.eval()

with torch.inference_mode():
    model_2_preds = model_2(x_test)

print(model_2_preds == preds)