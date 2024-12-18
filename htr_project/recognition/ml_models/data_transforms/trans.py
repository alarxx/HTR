#====================================================================
# Data transformations and Augmentation
# ImageFolder Dataset содержит PIL.Image объекты
#
# https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
#
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
#====================================================================

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class MinMaxWidth:
    def __init__(self, min_width=8, max_width=1024):
        self.min_width = min_width
        self.max_width = max_width

    def __call__(self, img):
        # img - это тензор [C, H, W]
        C, H, W = img.shape
        if W < self.min_width:
            new_img = torch.zeros((C, H, self.min_width))
            new_img[:, :, :W] = img
            return new_img
        if W > self.max_width:
            new_img = torch.zeros((C, H, self.max_width))
            new_img[:, :, :W] = img
            return new_img
        else:
            return img


# Кастомная трансформация для добавления случайного шума
class AddRandomNoise:
    def __init__(self, amount=0.05):
        self.amount = amount  # Максимальная величина шума

    def __call__(self, tensor):
        noise = (torch.rand(tensor.size()) - 0.5) * 2 * self.amount
        noisy_tensor = tensor + noise
        # Обрезаем значения до [0, 1]
        return torch.clamp(noisy_tensor, 0., 1.)


class DataTransforms:
    def __init__(self):
        pass