import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sns.set(style="darkgrid")
import copy
import itertools
import os
import pathlib

import splitfolders
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary
from torchvision import utils
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm, trange

warnings.filterwarnings("ignore")

labels_df = pd.read_csv("/content/drive/MyDrive/tumor_dataset/metadata.csv")
print(labels_df.head().to_markdown())

os.listdir("/content/drive/MyDrive/tumor_dataset/Brain Tumor Data Set/Brain Tumor Data Set")

data_dir = "/content/drive/MyDrive/tumor_dataset/Brain Tumor Data Set/Brain Tumor Data Set"
data_dir = pathlib.Path(data_dir)

# splitting dataset to train_set, val_set and test_set
splitfolders.ratio(data_dir, output="brain", seed=20, ratio=(0.8, 0.2))

# new dataset path
data_dir = "/content/brain"
data_dir = pathlib.Path(data_dir)

# define transformations
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define an object of the custom dataset for the train and validation.
train_set = torchvision.datasets.ImageFolder(data_dir.joinpath("train"), transform=transform)
train_set.transform
val_set = torchvision.datasets.ImageFolder(data_dir.joinpath("val"), transform=transform)
val_set.transform

# Visualization some images from train set
CLA_label = {0: "Brain Tumor", 1: "Healthy"}
fig = plt.figure(figsize=(10, 10))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_set), size=(1,)).item()
    img, label = train_set[sample_idx]
    fig.add_subplot(rows, cols, i)
    plt.title(CLA_label[label])
    plt.axis("off")
    img_np = img.numpy().transpose((1, 2, 0))
    # Clip pixel values to [0, 1]
    img_valid_range = np.clip(img_np, 0, 1)
    plt.imshow(img_valid_range)
    plt.suptitle("Brain Images", y=0.95)
plt.show()
