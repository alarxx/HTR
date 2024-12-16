import os
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted  # Пакет для естественной сортировки

# Путь к основному каталогу с папками
base_path = './'  # Замените на свой путь

# Получаем список папок, сортируем в алфавитном порядке
folders = natsorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

# Параметры сетки
rows, cols = 7, 6  # Укажите нужное количество строк и столбцов
fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
axs = axs.flatten()  # Плоский массив для упрощения доступа к ячейкам

# Обрабатываем каждую папку и добавляем изображение в сетку
for idx, folder in enumerate(folders):
    # Получаем путь к первому изображению в папке
    folder_path = os.path.join(base_path, folder)
    images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        # Открываем первое изображение в папке
        img_path = os.path.join(folder_path, images[0])
        img = Image.open(img_path).convert('RGB')

        # Показываем изображение в сетке
        axs[idx].imshow(img)
        axs[idx].set_title(folder)  # Название папки как заголовок
        axs[idx].axis('off')
    else:
        axs[idx].axis('off')  # Отключаем пустую ячейку

# Убираем лишние пустые ячейки, если папок меньше чем rows*cols
for ax in axs[len(folders):]:
    ax.axis('off')

plt.tight_layout()
plt.show()

