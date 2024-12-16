import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from utils import convert

from PIL import Image

from data_transforms.trans import DataTransforms


def show_aug(img, aug):
    # Визуализируем оригинальное и аугментированное изображение
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(aug)
    ax[1].set_title("Augmented Image")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()



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


def show_augmented_grid(img, transform, n_rows=6, n_cols=6):
    """
    Визуализирует оригинальное изображение и трансформированные версии в сетке (4x4).
    :param img: Оригинальное изображение (PIL.Image).
    :param transform: Трансформация (torchvision.transforms.Compose).
    :param n_rows: Количество строк в сетке.
    :param n_cols: Количество столбцов в сетке.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))

    # Первое изображение — оригинал
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Генерируем трансформированные изображения
    for i in range(n_rows):
        for j in range(n_cols):
            if i == 0 and j == 0:
                continue  # Пропускаем первое изображение (оригинал)

            # Применяем трансформацию
            transformed_tensor = transform(img)

            # Денормализуем, если нужно
            # denorm_tensor = denormalize(transformed_tensor)
            denorm_tensor = transformed_tensor

            # Преобразуем тензор в NumPy формат
            np_image = denorm_tensor.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            np_image = np.clip(np_image, 0, 1)  # Обрезаем значения в диапазон [0, 1]

            # Визуализируем изображение
            axes[i, j].imshow(np_image)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    path = "./data_transforms/0.png"

    image = torchvision.io.read_image(path)
    # print(type(image), image.dtype)
    # print(image.shape)
    # print()
    #
    # # torchvision.transforms принимает либо PIL.Image, либо torch.Tensor [C, H, W]
    # # Я бы никогда не хотел пользовать PIL.Image [W, H, C] - inconvenient and old library of 2009
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(p=1)
    # ])
    #
    # augmented_image = transform(image)
    #
    # print(type(augmented_image), augmented_image.dtype)
    # print(augmented_image.shape)
    # print()
    #
    #
    image = convert.tensor2cvmat(image)
    # augmented_image = convert.tensor2cvmat(augmented_image)
    #
    # show_aug(image, augmented_image)


    # https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
    print("Let's try PIL.Image now")

    data_transforms_obj = DataTransforms()
    transform = data_transforms_obj.train_transform

    img = Image.open(path).convert('RGB')
    img_tensor = transform(img)
    print(type(img_tensor), img_tensor.dtype)
    print(img_tensor.shape)
    print()

    # augmented_image = denormalize(img_tensor)
    augmented_image = img_tensor
    augmented_image = convert.ntensor2cvmat(augmented_image)

    show_aug(image, augmented_image)


    # Визуализируем 4x4 сетку изображений
    show_augmented_grid(img, transform)



