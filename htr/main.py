import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# Пример использования: предсказание для одного изображения
from PIL import Image

test_image_path = './printed_text.jpg'
img = Image.open(test_image_path).convert('RGB')
# img_tensor = val_transform(img).unsqueeze(0).to(device)
print(type(img))

