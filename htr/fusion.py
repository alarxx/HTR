import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from collections import Counter

from data_transforms.trans import DataTransforms
from classificator.cnns import FullyCNN10

from utils.io import Savior

import time


def display_class_percentages(labels, class_names, dataset_name="Dataset"):
    """
    Displays the percentage distribution of classes in a dataset as a bar plot.

    Args:
        labels (list): List of class labels for the dataset.
        class_names (list): List of class names corresponding to label indices.
        dataset_name (str): Name of the dataset (e.g., 'Train', 'Validation').
    """
    # Подсчитываем количество каждого класса
    class_counts = Counter(labels)
    class_counts = dict(sorted(class_counts.items()))
    # print("class_counts: ", class_counts)
    # print("class_counts.items()", class_counts.items())

    total_samples = len(labels)

    # Вычисляем процентное соотношение
    percentages = {
        class_names[label]: count  # (count / total_samples) * 100
        for label, count in class_counts.items()
    }

    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.bar(percentages.keys(), percentages.values())
    plt.xlabel("Classes")
    plt.ylabel("Percentage")
    plt.title(f"{dataset_name} Class Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ====================================================================
# Device (CPU/GPU/MPS)
# ====================================================================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)
print()


# ====================================================================
# Simple Dataset wrapper class for applying transforms
#
# Почему нужен этот класс?
# Мы хотим применять transforms на объекты Dataset (PIL.Image).
# Мы можем альтернативно сразу при чтении с ImageFolder применять transforms, но это не лаконичный подход.
# Проблема в том, что мы хотим чтобы на train выборке применялось Data Augmentation, а на validation оставляем.
# ====================================================================
class CustomDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # print("__getitem__ called:", img, label)
        if self.transform:
            img = self.transform(img)
        return img, label  # torch.Tensor values in [0, 1], torch.float32, torch.Size([C, H, W])


# ====================================================================
# Load Dataset
# Assumed structure:
# root_dir/class1/*.jpg
# root_dir/class2/*.png
# ...
#
# Every sub-directory is a distinct class (a letter of alphabet, in our case)
#
# ImageFolder inherits from DatasetFolder
#   transform can take transforms, but in our case we don't need it, we'll transform later
#   if allow_empty=True raises error if some of the sub-directories is empty
#
# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder
# ====================================================================

full_dataset = ImageFolder(root='./datasets/CMNIST', transform=None)  # -> Dataset - List[Tuple[<PIL.Image>, int]]

print("Full Dataset:", full_dataset)
print()
# Output: Dataset ImageFolder: Number of datapoints: 14, Root location: ./dataset
print("full_dataset[0]:", full_dataset[0])
print("full_dataset[0][1]:", type(full_dataset[0][1]))
# Output: (<PIL.Image.Image image mode=RGB size=41x56 at 0x7EFC13DBFF40>, 0)
print("Classes:", full_dataset.classes)  # list: [unique class_names]
# print("Targets:", full_dataset.targets) # list: [class_index1, class_index1, class_index2, class_index3...]
print("Class to index mapping:", full_dataset.class_to_idx)  # dict: {class_name: index}
print()

class_mapping = {  # Пока не используется
    "00": "А", "01": "Ә", "02": "Б", "03": "В", "04": "Г", "05": "Ғ",
    "06": "Д", "07": "Е", "08": "Ё", "09": "Ж", "10": "З", "11": "И",
    "12": "Й", "13": "К", "14": "Қ", "15": "Л", "16": "М", "17": "Н",
    "18": "Ң", "19": "О", "20": "Ө", "21": "П", "22": "Р", "23": "С",
    "24": "Т", "25": "У", "26": "Ұ", "27": "Ү", "28": "Ф", "29": "Х",
    "30": "Һ", "31": "Ц", "32": "Ч", "33": "Ш", "34": "Щ", "35": "Ъ",
    "36": "Ы", "37": "І", "38": "Ь", "39": "Э", "40": "Ю", "41": "Я"
}

mapped_labels = [class_mapping[class_id] for class_id in full_dataset.classes]
print("mapped_labels:", mapped_labels)

# display_class_percentages(full_dataset.targets, mapped_labels, dataset_name="Original")


# ====================================================================
# Train - Test Split
#
# Разделяем full_dataset на обучающую и тестовую выборки
# Cross-validation используем на global_train_dataset
#
# ====================================================================
train_indices, test_indices = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.2,  # 20% для теста
    random_state=42,
    stratify=[label for _, label in full_dataset]  # Учитываем баланс классов
)

# Создаем Subset для обучающей и тестовой выборок
global_train_dataset = Subset(full_dataset, train_indices)
global_test_dataset = Subset(full_dataset, test_indices)


print(f"Train dataset: {global_train_dataset}, {len(train_indices)}")
print(f"Test dataset: {global_test_dataset}, {len(test_indices)}")

# Убеждаюсь, что соотношения количества примеров в классах остаются одинаковыми.
# # Получаем лейблы для обучающей и тестовой выборок
global_train_labels = [full_dataset.targets[idx] for idx in train_indices]
global_test_labels = [full_dataset.targets[idx] for idx in test_indices]

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data_transforms.trans import DataTransforms
from classificator.cnns import FullyCNN10
from utils.io import Savior

# Параметры
model_prefix = "Alphabet_FCNN_10"
num_folds = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================
# Загрузка моделей
# ====================
def load_models(prefix, num_folds, device):
    models = []
    for i in range(num_folds):
        checkpoint = torch.load(f"{prefix}_{i}.pth", map_location=device)
        model = FullyCNN10(num_classes=len(checkpoint['classes'])).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"Loaded model {prefix}_{i}.pth")
    return models, checkpoint['classes']

# ====================
# Прогон через тестовый датасет
# ====================
def predict_with_models(models, dataloader, device):
    """
    Прогоняет тестовый датасет через несколько моделей и усредняет вероятности.
    """
    all_probs = []  # Список для хранения вероятностей всех моделей
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            avg_probs = None  # Сумма вероятностей по всем моделям

            for model in models:
                outputs = model(images)  # [batch_size, num_classes]
                probs = torch.softmax(outputs, dim=1)  # Преобразование в вероятности

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs  # Суммируем вероятности

            avg_probs /= len(models)  # Усредняем вероятности
            all_probs.append(avg_probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Конвертируем результаты
    all_probs = np.vstack(all_probs)  # Объединяем батчи
    final_preds = np.argmax(all_probs, axis=1)  # Берем класс с максимальной вероятностью
    return final_preds, all_labels

# ====================
# Подготовка тестового датасета
# ====================
test_ds = CustomDataset(global_test_dataset, transform=DataTransforms().val_transform)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# Загрузка моделей
models, class_names = load_models(model_prefix, num_folds, device)

start_time = time.time()

# Предсказания
final_preds, y_true = predict_with_models(models, test_loader, device)

print(f"Время выполнения кода: {time.time() - start_time:.5f} секунд")

# Метрики
test_accuracy = accuracy_score(y_true, final_preds)
test_f1 = f1_score(y_true, final_preds, average='macro')

print(f"Ensemble Test Accuracy: {test_accuracy:.4f}")
print(f"Ensemble Test F1-Score: {test_f1:.4f}")


# =======================
# Confusion Matrix
# =======================
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, final_preds), display_labels=mapped_labels) # это список имен классов в правильном порядке

plt.figure(figsize=(42, 42))
plt.rcParams.update({'font.size': 8})  # Общий размер шрифта
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix on Test Dataset")
plt.show()