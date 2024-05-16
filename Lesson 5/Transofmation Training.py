import pathlib
from typing import Tuple, Dict, List

import torch
import requests
import zipfile
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

	# 2. Initialize with a targ_dir and transform (optional) parameter
	def __init__(self, targ_dir: str, transform=None) -> None:

		# 3. Create class attributes
		# Get all image paths
		self.paths = list(pathlib.Path(targ_dir).glob(
			"*/*.jpg"))  # note: you'd have to update this if you've got .png's or .jpeg's
		# Setup transforms
		self.transform = transform
		# Create classes and class_to_idx attributes
		self.classes, self.class_to_idx = find_classes(targ_dir)

	# 4. Make function to load images
	def load_image(self, index: int) -> Image.Image:
		"Opens an image via a path and returns it."
		image_path = self.paths[index]
		return Image.open(image_path)

	# 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
	def __len__(self) -> int:
		"Returns the total number of samples."
		return len(self.paths)

	# 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
	def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
		"Returns one sample of data, data and label (X, y)."
		img = self.load_image(index)
		class_name = self.paths[index].parent.name  # expects path in data_folder/class_name/image.jpeg
		class_idx = self.class_to_idx[class_name]

		# Transform if necessary
		if self.transform:
			return self.transform(img), class_idx  # return data, label (X, y)
		else:
			return img, class_idx  # return data, label (X, y)


def walk_through_dir(dir_path):
	"""
		Walks through dir_path returning its contents.
		Args:
		  dir_path (str or pathlib.Path): target directory

		Returns:
		  A print out of:
			number of subdiretories in dir_path
			number of images (files) in each subdirectory
			name of each subdirectory
		"""
	for dirpath, dirnames, filenames in os.walk(dir_path):
		print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def plot_transformed_images(image_paths, transform, n=3, seed=42):
	"""Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
	random.seed(seed)
	random_image_paths = random.sample(image_paths, k=n)
	for image_path in random_image_paths:
		with Image.open(image_path) as f:
			fig, ax = plt.subplots(1, 2)
			ax[0].imshow(f)
			ax[0].set_title(f"Original \nSize: {f.size}")
			ax[0].axis("off")

			# Transform and plot image
			# Note: permute() will change shape of image to suit matplotlib
			# (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
			transformed_image = transform(f).permute(1, 2, 0)
			ax[1].imshow(transformed_image)
			ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
			ax[1].axis("off")

			fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
	"""Finds the class folder names in a target directory.

		Assumes target directory is in standard image classification format.

		Args:
			directory (str): target directory to load classnames from.

		Returns:
			Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

		Example:
			find_classes("food_images/train")
		"""
	# 1. Get the class names by scanning the target directory
	classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

	# 2. Raise an error if class names not found
	if not classes:
		raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

	# 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
	class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
	return classes, class_to_idx


# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
	# 2. Adjust display if n too high
	if n > 10:
		n = 10
		display_shape = False
		print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

	# 3. Set random seed
	if seed:
		random.seed(seed)

	# 4. Get random sample indexes
	random_samples_idx = random.sample(range(len(dataset)), k=n)

	# 5. Setup plot
	plt.figure(figsize=(16, 8))

	# 6. Loop through samples and display random samples
	for i, targ_sample in enumerate(random_samples_idx):
		targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

		# 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
		targ_image_adjust = targ_image.permute(1, 2, 0)

		# Plot adjusted samples
		plt.subplot(1, n, i + 1)
		plt.imshow(targ_image_adjust)
		plt.axis("off")
		if classes:
			title = f"class: {classes[targ_label]}"
			if display_shape:
				title = title + f"\nshape: {targ_image_adjust.shape}"
		plt.title(title)

	plt.show()


if __name__ == '__main__':
	# Setup path to data folder

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

	# Set seed
	random.seed(42)  # <- try changing this and see what happens

	# 1. Get all image paths (* means "any combination")
	image_path_list = list(image_path.glob("*/*/*.jpg"))

	# 2. Get random image path
	random_image_path = random.choice(image_path_list)

	# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
	image_class = random_image_path.parent.stem

	# 4. Open image
	img = Image.open(random_image_path)

	# 5. Print metadata
	print(f"Random image path: {random_image_path}")
	print(f"Image class: {image_class}")
	print(f"Image height: {img.height}")
	print(f"Image width: {img.width}")

	# Turn the image into an array
	img_as_array = np.asarray(img)

	# Plot the image with matplotlib
	plt.figure(figsize=(10, 7))
	plt.imshow(img_as_array)
	plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
	plt.axis(False)
	plt.show()

	# Write transform for image
	data_transform = transforms.Compose([
		# Resize the images to 64x64
		transforms.Resize(size=(64, 64)),
		# Flip the images randomly on the horizontal
		transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
		# Turn the image into a torch.Tensor
		transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
	])

	plot_transformed_images(image_path_list, transform=data_transform, n=3)

	# Use ImageFolder to create dataset(s)
	train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
	                                  transform=data_transform,  # transforms to perform on data (images)
	                                  target_transform=None)  # transforms to perform on labels (if necessary)

	test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

	print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

	class_names = train_data.classes
	class_dict = train_data.class_to_idx

	img, label = train_data[0][0], train_data[0][1]
	print(f"Image tensor:\n{img}")
	print(f"Image shape: {img.shape}")
	print(f"Image datatype: {img.dtype}")
	print(f"Image label: {label}")
	print(f"Label datatype: {type(label)}")

	# Rearrange the order of dimensions
	img_permute = img.permute(1, 2, 0)

	# Print out different shapes (before and after permute)
	print(f"Original shape: {img.shape} -> [color_channels, height, width]")
	print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

	# Turn train and test Datasets into DataLoaders
	train_dataloader = DataLoader(dataset=train_data,
	                              batch_size=1,  # how many samples per batch?
	                              num_workers=1,  # how many subprocesses to use for data loading? (higher = more)
	                              shuffle=True)  # shuffle the data?

	test_dataloader = DataLoader(dataset=test_data,
	                             batch_size=1,
	                             num_workers=1,
	                             shuffle=False)  # don't usually need to shuffle testing data

	img, label = next(iter(train_dataloader))

	# Batch size will now be 1, try changing the batch_size parameter above and see what happens
	print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
	print(f"Label shape: {label.shape}")

	# Setup path for target directory
	target_directory = train_dir
	print(f"Target directory: {target_directory}")

	# Get the class names from the target directory
	class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / "train"))])
	print(f"Class names found: {class_names_found}")

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

	train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
	test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

	display_random_images(train_data, n=5, classes=class_names, seed=None)

	# Turn train and test custom Dataset's into DataLoader's
	train_dataloader_custom = DataLoader(dataset=train_data_custom,  # use custom created train Dataset
	                                     batch_size=1,  # how many samples per batch?
	                                     num_workers=0,
	                                     # how many subprocesses to use for data loading? (higher = more)
	                                     shuffle=True)  # shuffle the data?

	test_dataloader_custom = DataLoader(dataset=test_data_custom,  # use custom created test Dataset
	                                    batch_size=1,
	                                    num_workers=0,
	                                    shuffle=False)  # don't usually need to shuffle testing data

	# Get image and label from custom DataLoader
	img_custom, label_custom = next(iter(train_dataloader_custom))

	# Batch size will now be 1, try changing the batch_size parameter above and see what happens
	print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
	print(f"Label shape: {label_custom.shape}")

	train_transforms = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.TrivialAugmentWide(num_magnitude_bins=31),  # how intense
		transforms.ToTensor()  # use ToTensor() last to get everything between 0 & 1
	])

	# Don't need to perform augmentation on the test data
	test_transforms = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
	])

	simple_transform = transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ToTensor(),
	])
