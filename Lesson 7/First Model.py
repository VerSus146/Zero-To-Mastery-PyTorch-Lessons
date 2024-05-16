import random

import torch
import torchvision
import matplotlib.pyplot as plt
import torch
import torchvision
import os
import zipfile

from helper_functions import plot_loss_curves
from pathlib import Path

import requests
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular.going_modular import data_setup, engine

device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if __name__ == '__main__':

	train_dir = image_path / "train"
	test_dir = image_path / "test"

	manual_transforms = transforms.Compose([
	    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
	    transforms.ToTensor(), # 2. Turn image values to between 0 & 1
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
	                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
	])

	train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
	                                                                               test_dir=test_dir,
	                                                                               transform=manual_transforms, # resize, convert images to between 0 & 1 and normalize them
	                                                                               batch_size=32) # set mini-batch size to 32

	weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
	auto_transforms = weights.transforms()

	train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
	                                                                               test_dir=test_dir,
	                                                                               transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
	                                                                               batch_size=32) # set mini-batch size to 32

	weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights
	model = torchvision.models.efficientnet_b0(weights=weights).to(device)

	for param in model.features.parameters():
		param.requires_grad = False

	# Set the manual seeds
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)

	# Get the length of class_names (one output unit for each class)
	output_shape = len(class_names)

	# Recreate the classifier layer and seed it to the target device
	model.classifier = torch.nn.Sequential(
	    torch.nn.Dropout(p=0.2, inplace=True),
	    torch.nn.Linear(in_features=1280,
	                    out_features=output_shape, # same number of output units as our number of classes
	                    bias=True)).to(device)

	summary(model,
	        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
	        verbose=0,
	        col_names=["input_size", "output_size", "num_params", "trainable"],
	        col_width=20,
	        row_settings=["var_names"]
	)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	# Set the random seeds
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)

	# Start the timer
	from timeit import default_timer as timer
	start_time = timer()

	# Setup training and save the results
	results = engine.train(model=model,
	                       train_dataloader=train_dataloader,
	                       test_dataloader=test_dataloader,
	                       optimizer=optimizer,
	                       loss_fn=loss_fn,
	                       epochs=5,
	                       device=device)

	# End the timer and print out how long it took
	end_time = timer()
	print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

	plot_loss_curves(results)

	from typing import List, Tuple

	from PIL import Image


	# 1. Take in a trained model, class names, image path, image size, a transform and target device
	def pred_and_plot_image(model: torch.nn.Module,
	                        image_path: str,
	                        class_names: List[str],
	                        image_size: Tuple[int, int] = (224, 224),
	                        transform: torchvision.transforms = None,
	                        device: torch.device = device):

		# 2. Open image
		img = Image.open(image_path)

		# 3. Create transformation for image (if one doesn't exist)
		if transform is not None:
			image_transform = transform
		else:
			image_transform = transforms.Compose([
				transforms.Resize(image_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
				                     std=[0.229, 0.224, 0.225]),
			])

		### Predict on image ###

		# 4. Make sure the model is on the target device
		model.to(device)

		# 5. Turn on model evaluation mode and inference mode
		model.eval()
		with torch.inference_mode():
			# 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
			transformed_image = image_transform(img).unsqueeze(dim=0)

			# 7. Make a prediction on image with an extra dimension and send it to the target device
			target_image_pred = model(transformed_image.to(device))

		# 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
		target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

		# 9. Convert prediction probabilities -> prediction labels
		target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

		# 10. Plot image with predicted label and probability
		plt.figure()
		plt.imshow(img)
		plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
		plt.axis(False);
		plt.show()

		# Get a random list of image paths from test set
		num_images_to_plot = 3
		test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))  # get list all image paths from test data
		test_image_path_sample = random.sample(population=test_image_path_list,
		                                       # go through all of the test image paths
		                                       k=num_images_to_plot)  # randomly select 'k' image paths to pred and plot

		# Make predictions on and plot the images
		for image_path in test_image_path_sample:
			pred_and_plot_image(model=model,
			                    image_path=image_path,
			                    class_names=class_names,
			                    # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
			                    image_size=(224, 224))

		# Download custom image
		import requests

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

		# Predict on custom image
		pred_and_plot_image(model=model,
		                    image_path=custom_image_path,
		                    class_names=class_names)