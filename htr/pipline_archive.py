import os
import re

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
from classificator.cnns import FullyCNN10, FullyCNN12

import pickle
from utils.io import Savior

from torchvision.transforms.functional import to_pil_image


def display_class_images(full_dataset, classes, num_cols=5):
    """
    Отображает по одному изображению на каждый класс из full_dataset.

    Args:
        full_dataset: Dataset, из которого берутся изображения и метки.
        classes: Список имен классов (full_dataset.classes).
        num_cols: Количество столбцов для отображения.
    """
    # Найдем индексы первого изображения для каждого класса
    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset):
        if label not in class_indices:
            class_indices[label] = idx
            # Если нашли все классы, прерываем цикл
            if len(class_indices) == len(classes):
                break

    # Настроим размеры для отображения
    num_classes = len(classes)
    num_rows = (num_classes + num_cols - 1) // num_cols  # количество строк

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    axes = axes.flatten()  # Разворачиваем массив осей для удобного доступа

    # Отобразим изображения по каждому классу
    for i, (label, idx) in enumerate(class_indices.items()):
        img, _ = full_dataset[idx]  # Получаем изображение и метку
        # img = to_pil_image(img)    # Конвертируем в PIL.Image
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"{classes[label]} (Label {label})")

    # Удалим лишние оси, если есть
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

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


# Класс с сортировкой, Bullshit
# def natural_sort_key(s):
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
# class SortingImageFolder(ImageFolder):
#     def find_classes(self, directory):
#         # Получаем список классов (папок)
#         classes = [d.name for d in os.scandir(directory) if d.is_dir()]
#         # Сортируем их естественным порядком
#         classes.sort(key=natural_sort_key)
#         class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#         return classes, class_to_idx

full_dataset = ImageFolder(root='./datasets/CMNIST', transform=None)  # -> Dataset - List[Tuple[<PIL.Image>, int]]
# full_dataset = SortingImageFolder(root='./datasets/CMNIST', transform=None)

print("Full Dataset:", full_dataset)
print()
# Output: Dataset ImageFolder: Number of datapoints: 14, Root location: ./dataset
print("full_dataset[0]:", full_dataset[0])
print("full_dataset[0]:", type(full_dataset[0][1]))
# Output: (<PIL.Image.Image image mode=RGB size=41x56 at 0x7EFC13DBFF40>, 0)
print("Classes:", full_dataset.classes)  # list: [class_names]
# print("Targets:", full_dataset.targets) # list: [class_names]
print("Class to index mapping:", full_dataset.class_to_idx)  # dict: {class_name: index}
print()

# Пример использования:
# display_class_images(full_dataset, full_dataset.classes)

# Разделяем full_dataset на обучающую и тестовую выборки
train_indices, test_indices = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.2,  # 20% для теста
    random_state=42,
    stratify=[label for _, label in full_dataset]  # Учитываем баланс классов
)

# Создаем Subset для обучающей и тестовой выборок
global_train_dataset = Subset(full_dataset, train_indices)
global_test_dataset = Subset(full_dataset, test_indices)

display_class_images(global_test_dataset, full_dataset.classes)

print(f"Train dataset: {global_train_dataset}")
print(f"Test dataset: {global_test_dataset}")

# Убеждаюсь, что соотношения количества примеров в классах остаются одинаковыми.
# # Получаем лейблы для обучающей и тестовой выборок
# train_labels = [full_dataset.targets[idx] for idx in train_indices]
# test_labels = [full_dataset.targets[idx] for idx in test_indices]
#
# # Подсчитываем количество каждого класса
# train_class_counts = Counter(train_labels)
# test_class_counts = Counter(test_labels)
#
# # Вычисляем процентное соотношение
# train_percentages = {
#     full_dataset.classes[label]: count / len(train_labels) * 100
#     for label, count in train_class_counts.items()
# }
# test_percentages = {
#     full_dataset.classes[label]: count / len(test_labels) * 100
#     for label, count in test_class_counts.items()
# }
#
# # Вывод результатов
# print("Train class percentages:")
# for class_name, percentage in train_percentages.items():
#     print(f"{class_name}: {percentage:.2f}%")
#
# print("\nTest class percentages:")
# for class_name, percentage in test_percentages.items():
#     print(f"{class_name}: {percentage:.2f}%")


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


# ====================================================================
# Functions for one training/validation epoch
# ====================================================================
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0  # sum of losses for the whole batch per epoch.
    correct = 0  # number of correct predictions.
    total = 0  # total number of samples (examples) in the epoch.

    y_true = []
    y_pred = []

    unique_labels = set()  # Для подсчёта уникальных лейблов

    i = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # model принимает [number, c, h, w], here number=batch_size
        outputs = model(images)
        loss = criterion(outputs, labels)  # average loss value, i.e. divided by number
        loss.backward()
        optimizer.step()

        # tensor.item() gives scalar value, Python Number
        running_loss += loss.item() * images.size(0)
        # https://pytorch.org/docs/main/generated/torch.max.html
        _, preds = torch.max(outputs, 1)  # mat -> vector, preds - indexes, _ - values
        correct += (preds == labels).sum().item()  # Boolean Tensor
        total += labels.size(0)  # len(vector), actually batch_size

        # Сохраняем метки, потому что для расчета F-Score нам нужны все TP, TN, FP, FN
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        unique_labels.update(labels.cpu().numpy())

        i += 1
        if i % 10 != 0:
            continue
        print(f"Train epoch, minibatch: {i},",
              f"Train Loss: {loss.item():.4f}, Train Acc: {((preds == labels).sum().item()) / labels.size(0):.4f}\n")

    print(f"Number of unique labels encountered: {len(unique_labels)}", unique_labels)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Multiclass F1-score (macro), one-to-many
    # Equalises the contribution of each class
    # We care about every class, no matter what size it is
    epoch_f1 = f1_score(y_true, y_pred, average='macro')

    return epoch_loss, epoch_acc, epoch_f1


def validate_one_epoch(model, dataloader, criterion):
    model.eval()  # for things like dropout
    running_loss = 0.0
    correct = 0
    total = 0

    y_true = list(range(len(full_dataset.classes)))
    y_pred = list(range(len(full_dataset.classes)))

    with torch.no_grad():
        i = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # print(f"Val epoch, minibatch: {i},",
            #        f"Val Loss: {loss.item():.4f}, Val Acc: {((preds == labels).sum().item())/labels.size(0):.4f}\n")

            i += 1

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(y_true, y_pred, average='macro')

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred),
                                  display_labels=mapped_labels)  # это список имен классов в правильном порядке
    plt.figure(figsize=(42, 42))
    plt.rcParams.update({'font.size': 8})  # Общий размер шрифта
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix on Validation Dataset")
    plt.show()

    return epoch_loss, epoch_acc, epoch_f1


# ====================================================================
# Hyperparameters
# ====================================================================
batch_size = 256
num_epochs = 4
learning_rate = 0.0001

# ====================================================================
# Cross-validation
# ====================================================================
indices = np.arange(len(global_train_dataset))  # [0, len]
labels = [label for _, label in global_train_dataset]
# print("indices", indices)
# print("labels", labels)
print()

num_folds = 2
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# ====================================================================
# Cross-validation Cycle
# ====================================================================
fold_results = []

# kf.split(indices) executes only once
# for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
    print("-" * 30)
    print(f"Fold {fold + 1}/{num_folds}")
    # print("val_idx, train_idx:", val_idx, train_idx)

    # Получаем лейблы классов
    train_labels = [labels[idx] for idx in train_idx]
    val_labels = [labels[idx] for idx in val_idx]
    # print("val_labels, train_labels:", val_labels, train_labels)

    # # Подсчитываем количество каждого класса
    # train_class_counts = Counter(train_labels)
    # val_class_counts = Counter(val_labels)
    # #
    # # Вычисляем процентное соотношение
    # train_percentages = {full_dataset.classes[label]: count / len(train_labels) * 100
    #                      for label, count in train_class_counts.items()}
    # val_percentages = {full_dataset.classes[label]: count / len(val_labels) * 100
    #                    for label, count in val_class_counts.items()}
    # #
    # # Вывод результатов
    # print("Train class percentages:")
    # for class_name, percentage in train_percentages.items():
    #     print(f"{class_name}: {percentage:.2f}%")
    # print()
    # #
    # print("\nValidation class percentages:")
    # for class_name, percentage in val_percentages.items():
    #     print(f"{class_name}: {percentage:.2f}%")
    # print()

    # Подготовка датасетов для конкретного фолда
    train_subset = Subset(global_train_dataset, train_idx)
    val_subset = Subset(global_train_dataset, val_idx)

    print("train_subset:", train_subset)
    print(f"Train dataset size: {len(train_subset)}")
    print("val_subset:", val_subset)
    print(f"Validation dataset size: {len(val_subset)}")
    print("-" * 30)

    train_ds = CustomDataset(train_subset, transform=DataTransforms().train_transform)
    val_ds = CustomDataset(val_subset, transform=DataTransforms().val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(full_dataset.classes)
    model = FullyCNN10(num_classes=num_classes).to(device)
    # class_counts = Counter(labels)
    # class_weights = torch.tensor([1.0 / class_counts[i] for i in range(len(class_counts))], device=device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_acc_history = []
    train_f1_history = []
    train_loss_history = []
    val_acc_history = []
    val_f1_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer,
                                                          criterion=criterion)
        val_loss, val_acc, val_f1 = validate_one_epoch(model=model, dataloader=val_loader, criterion=criterion)

        print(f"Epoch [{epoch + 1}/{num_epochs}]\n",
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F-Score: {train_f1:.4f}\n",
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F-Score: {val_f1:.4f}")

        # Высчитываем точность на тестовом датасете без аугментации для честности сравнения между train и validation accuracy
        _, train_acc, train_f1 = validate_one_epoch(model=model, dataloader=train_loader, criterion=criterion)

        train_acc_history.append(train_acc)
        train_f1_history.append(train_f1)
        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)
        val_f1_history.append(val_f1)
        val_loss_history.append(val_loss)

    # Сохраняем результаты фолда
    # fold_results.append((train_acc_history, train_f1_history, val_acc_history, val_f1_history, model.state_dict()))
    fold_results.append((train_acc_history, train_f1_history, train_loss_history, val_acc_history, val_f1_history,
                         val_loss_history, model.state_dict()))

# Теперь нужно визуализировать среднюю точность по всем фолдам на каждой эпохе на обучающей выборке и на валидационной.

model_prefix = "Alphabet_FCNN_5"
savior = Savior()
savior.save_models(fold_results, full_dataset, prefix=model_prefix)

# Initialize accumulators for train and validation accuracies
avg_train_acc = np.zeros(num_epochs)
avg_val_acc = np.zeros(num_epochs)

# Accumulate accuracies across folds
for fold_result in fold_results:
    train_acc_history, train_f1_history, train_loss_history, val_acc_history, val_f1_history, val_loss_history, _ = fold_result
    avg_train_acc += np.array(train_acc_history)
    avg_val_acc += np.array(val_acc_history)

# Compute average accuracies
avg_train_acc /= len(fold_results)
avg_val_acc /= len(fold_results)

# Visualize the average accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), avg_train_acc, label='Train Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), avg_val_acc, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Average Train and Validation Accuracy Across Folds')
plt.legend()
plt.grid(True)
plt.show()

# =======================
# Выбор лучшей модели
# =======================
best_fold = -1
best_val_acc = -1

for i, fold_result in enumerate(fold_results):
    val_acc_last_epoch = fold_result[4][-1]  # val_acc_history на последней эпохе, ! changed to f-score 2->3
    if val_acc_last_epoch > best_val_acc:
        best_val_acc = val_acc_last_epoch
        best_fold = i

print(f"Best model is from fold {best_fold + 1} with Validation Accuracy: {best_val_acc:.4f}")

# =======================
# Загрузка лучшей модели
# =======================
checkpoint = torch.load(f"{model_prefix}_{best_fold}.pth", map_location=device)
model = FullyCNN10(num_classes=len(checkpoint['classes'])).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Best model loaded successfully!")

# =======================
# Тестирование на global_test_dataset
# =======================
test_ds = CustomDataset(global_test_dataset, transform=DataTransforms().val_transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# =======================
# Расчёт метрик
# =======================
test_accuracy = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred, average='macro')
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

# =======================
# Confusion Matrix
# =======================


disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred),
                              display_labels=mapped_labels)  # это список имен классов в правильном порядке

plt.figure(figsize=(42, 42))
plt.rcParams.update({'font.size': 8})  # Общий размер шрифта
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix on Test Dataset")
plt.show()