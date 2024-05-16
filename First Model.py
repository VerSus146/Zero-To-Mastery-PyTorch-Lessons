import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

what_were_covering = {1: "data (prepare and load)",
                      2: "build model",
                      3: "fitting the model to data (training)",
                      4: "making predictions and evaluating a model (inference)",
                      5: "saving and loading a model",
                      6: "putting it all together"
                      }

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(what_were_covering.get(1))

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_valid,
                     test_labels=y_valid,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    # Plot data into tables

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x * self.weights


print(what_were_covering[1])

torch.manual_seed(42)

model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(X_valid)

print(f"Number of testing samples: {len(X_valid)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 100

train_loss_values = []
test_loss_values = []
epoch_count = []

print(what_were_covering[2])

for epoch in range(epochs):
    ###Training

    # Training mode
    model_0.train()

    # Forward pass
    y_pred = model_0.forward(X_train)

    # Calcule loss against known training data
    loss = loss_fn(y_pred, y_train)

    # Zero Grad on optimizer
    optimizer.zero_grad()

    # Loss backward step
    loss.backward()

    # Optimizer step
    optimizer.step()

    ###Test

    model_0.eval()

    print(what_were_covering[3])

    with torch.inference_mode():
        test_pred = model_0(X_valid)

        test_loss = loss_fn(test_pred, y_valid.type(torch.float32))

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# Set model to eval mode
model_0.eval()

with torch.inference_mode():
    y_preds = model_0(X_valid)

plot_predictions(predictions=y_preds)

print(what_were_covering[4])

# Saving Model

# Create Folder
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create Path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save Model
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),  # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

# Loading Models

# As we've only saved the state_dict() - State of the model - we have to create a new empty model and load the
# Data onto it
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_valid)  # perform a forward pass on the test data with the loaded model

# Compare previous model predictions with loaded model predictions (these should be the same)
print(y_preds == loaded_model_preds)
