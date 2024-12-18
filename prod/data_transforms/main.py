import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from utils import convert

from PIL import Image


def show_aug(img, aug):
    # Визуализируем оригинальное и аугментированное изображение
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(augmented_image)
    ax[1].set_title("Augmented Image")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


# Кастомная трансформация для паддинга до 64x64 и ресайза только при необходимости
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


def show_augmented_grid(img, transform, n_rows=4, n_cols=4):
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
            denorm_tensor = denormalize(transformed_tensor)

            # Преобразуем тензор в NumPy формат
            np_image = denorm_tensor.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            np_image = np.clip(np_image, 0, 1)  # Обрезаем значения в диапазон [0, 1]

            # Визуализируем изображение
            axes[i, j].imshow(np_image)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    # path = "./printed_text.jpg"
    path = "./0.png"

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



    print("Let's try PIL.Image now")

    transform = transforms.Compose([
        transforms.Resize(64),
        # transforms.RandomResizedCrop(64, scale=(0.5, 1.5)), # Рандомное увеличение/уменьшение
        # transforms.RandomRotation(10), # Повороты
        transforms.RandomAffine(
            degrees=0,  # Повороты на ± градусов
            translate=(0.2, 0.2),  # Сдвиг до % от ширины/высоты
            scale=(0.5, 2),  # Масштабирование
            shear=30  # Наклон на ± градусов
        ),
        # transforms.RandomHorizontalFlip(p=1),
        # PadToSquareAndConditionalResize(64), # Паддинг или ресайз до 64x64
        transforms.ToTensor(),
        AddRandomNoise(amount=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(path).convert('RGB')
    img_tensor = transform(img)
    print(type(img_tensor), img_tensor.dtype)
    print(img_tensor.shape)
    print()

    augmented_image = denormalize(img_tensor)
    augmented_image = convert.ntensor2cvmat(augmented_image)

    show_aug(image, augmented_image)


    # Визуализируем 4x4 сетку изображений
    show_augmented_grid(img, transform)



