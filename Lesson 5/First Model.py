import torch
import requests
import zipfile
import os
import torchvision

import torch.nn as nn

from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Dict, List

from timeit import default_timer as timer
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


class TinyVGG(nn.Module):
	"""
	Model architecture copying TinyVGG from:
	https://poloclub.github.io/cnn-explainer/
	"""

	def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
		super().__init__()
		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.conv_block_2 = nn.Sequential(
			nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			# Where did this in_features shape come from?
			# It's because each layer of our network compresses and changes the shape of our inputs data.
			nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape)
		)

	def forward(self, x: torch.Tensor):
		return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
	model.train()

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

		# Calculate and accumulate accuracy metric across all batches
		y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
		train_acc += (y_pred_class == y).sum().item() / len(y_pred)

	# Adjust metrics to get average loss and accuracy per batch
	train_loss = train_loss / len(dataloader)
	train_acc = train_acc / len(dataloader)
	return train_loss, train_acc


def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
	model.eval()

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
			test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

	# Adjust metrics to get average loss and accuracy per batch
	test_loss = test_loss / len(dataloader)
	test_acc = test_acc / len(dataloader)
	return test_loss, test_acc


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs: int = 5):
	results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

	# 3. Loop through training and testing steps for a number of epochs
	for epoch in tqdm(range(epochs)):
		train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
		test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)

		# 4. Print out what's happening
		print(
			f"Epoch: {epoch + 1} | "
			f"train_loss: {train_loss:.4f} | "
			f"train_acc: {train_acc:.4f} | "
			f"test_loss: {test_loss:.4f} | "
			f"test_acc: {test_acc:.4f}"
		)

		# 5. Update results dictionary
		results["train_loss"].append(train_loss)
		results["train_acc"].append(train_acc)
		results["test_loss"].append(test_loss)
		results["test_acc"].append(test_acc)

	# 6. Return the filled results at the end of the epochs
	return results


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
	plt.show()


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = device):
	"""Makes a prediction on a target image and plots the image with its prediction."""

	# 1. Load in image and convert the tensor values to float32
	target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

	# 2. Divide the image pixel values by 255 to get them between [0, 1]
	target_image = target_image / 255.

	# 3. Transform if necessary
	if transform:
		target_image = transform(target_image)

	# 4. Make sure the model is on the target device
	model.to(device)

	# 5. Turn on model evaluation mode and inference mode
	model.eval()
	with torch.inference_mode():
		# Add an extra dimension to the image
		target_image = target_image.unsqueeze(dim=0)

		# Make a prediction on image with an extra dimension and send it to the target device
		target_image_pred = model(target_image.to(device))

	# 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
	target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

	# 7. Convert prediction probabilities -> prediction labels
	target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

	# 8. Plot the image alongside the prediction and prediction probability
	plt.imshow(target_image.squeeze().permute(1, 2, 0))  # make sure it's the right size for matplotlib
	if class_names:
		title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
	else:
		title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
	plt.title(title)
	plt.axis(False)
	plt.show()


if __name__ == '__main__':

	data_path = Path("data/")
	image_path = data_path / "pizza_steak_sushi"

	# If the image folder doesn't exist, download it and prepare it...
	if image_path.is_dir():
		print(f"{image_path} directory exists.")
	else:
		print(f"Did not find {image_path} directory, creating one...")
		image_path.mkdir(parents=True, exist_ok=True)
		# Download pizza, steak, sushi data
		with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
			request = requests.get(
				"https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
			print("Downloading pizza, steak, sushi data...")
			f.write(request.content)

		# Unzip pizza, steak, sushi data
		with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
			print("Unzipping pizza, steak, sushi data...")
			zip_ref.extractall(image_path)

	train_dir = image_path / "train"
	test_dir = image_path / "test"

	# Create simple transform
	simple_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), ])

	train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
	test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

	# Setup batch size and number of workers
	BATCH_SIZE = 32
	NUM_WORKERS = os.cpu_count()

	# Create DataLoader's
	train_dataloader_simple = DataLoader(train_data_simple, batch_size=BATCH_SIZE, shuffle=True,
	                                     num_workers=NUM_WORKERS)

	test_dataloader_simple = DataLoader(test_data_simple, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	data_transform = transforms.Compose([
		# Resize the images to 64x64
		transforms.Resize(size=(64, 64)),
		# Flip the images randomly on the horizontal
		transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
		# Turn the image into a torch.Tensor
		transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
	])

	train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
	test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

	class_names = train_data.classes

	torch.manual_seed(42)
	torch.cuda.manual_seed(42)

	# Set number of epochs
	NUM_EPOCHS = 5

	# Recreate an instance of TinyVGG
	model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)

	# Setup loss function and optimizer
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

	# Create training transform with TrivialAugment
	train_transform_trivial_augment = transforms.Compose([transforms.Resize((64, 64)),
	                                  transforms.TrivialAugmentWide(num_magnitude_bins=31), transforms.ToTensor()])

	# Create testing transform (no data augmentation)
	test_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

	train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
	test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

	# Setup custom image path
	custom_image_path = data_path / "04-pizza-dad.jpeg"

	# Download the image if it doesn't already exist
	if not custom_image_path.is_file():
		with open(custom_image_path, "wb") as f:
			# When downloading from GitHub, need to use the "raw" file link
			request = requests.get(
				"https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
			print(f"Downloading {custom_image_path}...")
			f.write(request.content)
	else:
		print(f"{custom_image_path} already exists, skipping download.")

	# Read in custom image
	custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))

	# Print out image data
	print(f"Custom image tensor:\n{custom_image_uint8}\n")
	print(f"Custom image shape: {custom_image_uint8.shape}\n")
	print(f"Custom image dtype: {custom_image_uint8.dtype}")

	# Load in custom image and convert the tensor values to float32
	custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

	# Divide the image pixel values by 255 to get them between [0, 1]
	custom_image = custom_image / 255.

	# Print out image data
	print(f"Custom image tensor:\n{custom_image}\n")
	print(f"Custom image shape: {custom_image.shape}\n")
	print(f"Custom image dtype: {custom_image.dtype}")

	# Create transform pipleine to resize image
	custom_image_transform = transforms.Compose([
		transforms.Resize((64, 64)),
	])

	# Transform target image
	custom_image_transformed = custom_image_transform(custom_image)

	# Print out original shape and new shape
	print(f"Original shape: {custom_image.shape}")
	print(f"New shape: {custom_image_transformed.shape}")

	model_0.eval()
	with torch.inference_mode():
		# Add an extra dimension to image
		custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)

		# Print out different shapes
		print(f"Custom image transformed shape: {custom_image_transformed.shape}")
		print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")

		# Make a prediction on image with an extra dimension
		custom_image_pred = model_0(custom_image_transformed.unsqueeze(dim=0).to(device))

		# Print out prediction logits
		print(f"Prediction logits: {custom_image_pred}")

		# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
		custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
		print(f"Prediction probabilities: {custom_image_pred_probs}")

		# Convert prediction probabilities -> prediction labels
		custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
		print(f"Prediction label: {custom_image_pred_label}")

		custom_image_pred_class = class_names[
			custom_image_pred_label.cpu()]  # put pred label to CPU, otherwise will error
