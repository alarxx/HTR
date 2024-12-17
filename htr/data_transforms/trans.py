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


# Кастомная трансформация для паддинга до 64x64 и ресайза только при необходимости
# Я перестал использовать это, потому что я использую Affine и Perspective, и этот класс просто лишняя головная боль.
class PadToSquareAndConditionalResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # Определяем размеры изображения
        w, h = image.size

        # Если изображение больше 64x64, делаем ресайз с сохранением пропорций
        if max(w, h) > self.size:
            scale = self.size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            w, h = image.size

        # Если изображение меньше 64x64, дополняем нулями до 64x64
        padded_image = Image.new("RGB", (self.size, self.size), (0, 0, 0))  # Черный фон
        padded_image.paste(image, ((self.size - w) // 2, (self.size - h) // 2))  # Центрируем

        return padded_image


# Кастомная трансформация для добавления случайного шума
class AddRandomNoise:
    def __init__(self, amount=0.05):
        self.amount = amount  # Максимальная величина шума

    def __call__(self, tensor):
        noise = (torch.rand(tensor.size()) - 0.5) * 2 * self.amount
        noisy_tensor = tensor + noise
        # Обрезаем значения до [0, 1]
        return torch.clamp(noisy_tensor, 0., 1.)


# Функция для денормализации
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Выполняет денормализацию тензора.
    :param tensor: нормализованный тензор PyTorch
    :param mean: список средних значений (по каналам)
    :param std: список стандартных отклонений (по каналам)
    :return: денормализованный тензор
    """
    mean = torch.tensor(mean).view(3, 1, 1)  # Преобразуем mean в тензор
    std = torch.tensor(std).view(3, 1, 1)    # Преобразуем std в тензор
    tensor = tensor * std + mean             # Денормализация
    return tensor



class DataTransforms:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            # transforms.RandomAffine(
            #     degrees=30,  # Повороты на ± градусов
            #     translate=(0.2, 0.2),  # Сдвиг до % от ширины/высоты
            #     scale=(0.5, 2),  # Масштабирование
            #     shear=30  # Наклон на ± градусов
            # ),
            # PadToSquareAndConditionalResize(64), # Паддинг или ресайз до 64x64
            transforms.ToTensor(),
            AddRandomNoise(amount=0.1),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
