import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from collections import Counter

from data_transforms.trans import DataTransforms
from classificator.cnns import FullyCNN10


#====================================================================
# Device (CPU/GPU/MPS)
#====================================================================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)
print()


#====================================================================
# Simple Dataset wrapper class for applying transforms
#
# Почему нужен этот класс?
# Мы хотим применять transforms на объекты Dataset (PIL.Image).
# Мы можем альтернативно сразу при чтении с ImageFolder применять transforms, но это не лаконичный подход.
# Проблема в том, что мы хотим чтобы на train выборке применялось Data Augmentation, а на validation оставляем.
#====================================================================
class CustomDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label # torch.Tensor values in [0, 1], torch.float32, torch.Size([C, H, W])


#====================================================================
# Functions for one training/validation epoch
#====================================================================
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0 # sum of losses for the whole batch per epoch.
    correct = 0 # number of correct predictions.
    total = 0 # total number of samples (examples) in the epoch.

    y_true = []
    y_pred = []

    i = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # model принимает [number, c, h, w], here number=batch_size
        outputs = model(images)
        loss = criterion(outputs, labels) # average loss value, i.e. divided by number
        loss.backward()
        optimizer.step()

        # tensor.item() gives scalar value, Python Number
        running_loss += loss.item() * images.size(0)
        # https://pytorch.org/docs/main/generated/torch.max.html
        _, preds = torch.max(outputs, 1) # mat -> vector, preds - indexes, _ - values
        correct += (preds == labels).sum().item() # Boolean Tensor
        total += labels.size(0) # len(vector), actually batch_size

        # Сохраняем метки, потому что для расчета F-Score нам нужны все TP, TN, FP, FN
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        i+=1
        if i % 10 != 0:
            continue
        print(f"Train epoch, minibatch: {i},",
              f"Train Loss: {loss.item():.4f}, Train Acc: {((preds == labels).sum().item())/labels.size(0):.4f}\n")

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Multiclass F1-score (macro), one-to-many
    # Equalises the contribution of each class
    # We care about every class, no matter what size it is
    epoch_f1 = f1_score(y_true, y_pred, average='macro')

    return epoch_loss, epoch_acc, epoch_f1


def validate_one_epoch(model, dataloader, criterion):
    model.eval() # for things like dropout
    running_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

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

            i+=1


    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(y_true, y_pred, average='macro')

    return epoch_loss, epoch_acc, epoch_f1



#====================================================================
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
#====================================================================
full_dataset = ImageFolder(root='./datasets/CMNIST', transform=None) # -> Dataset - List[Tuple[<PIL.Image>, int]]

print("Full Dataset:", full_dataset)
print()
# Output: Dataset ImageFolder: Number of datapoints: 14, Root location: ./dataset
print("full_dataset[0]:", full_dataset[0])
# Output: (<PIL.Image.Image image mode=RGB size=41x56 at 0x7EFC13DBFF40>, 0)
print("Classes:", full_dataset.classes) # list: [class_names]
# print("Targets:", full_dataset.targets) # list: [class_names]
print("Class to index mapping:", full_dataset.class_to_idx) # dict: {class_name: index}
print()


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

#====================================================================
# Hyperparameters
#====================================================================
batch_size = 128
num_epochs = 2
learning_rate = 0.001


#====================================================================
# Cross-validation
#====================================================================
indices = np.arange(len(global_train_dataset)) # [0, len]
labels = [label for _, label in global_train_dataset]
# print("indices", indices)
# print("labels", labels)
print()

num_folds = 3
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)


#====================================================================
# Cross-validation Cycle
#====================================================================
fold_results = []

# kf.split(indices) executes only once
# for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
    print("-" * 30)
    print(f"Fold {fold+1}/{num_folds}")
    # print("val_idx, train_idx:", val_idx, train_idx)

    # Получаем лейблы классов
    train_labels = [labels[idx] for idx in train_idx]
    val_labels = [labels[idx] for idx in val_idx]
    # print("val_labels, train_labels:", val_labels, train_labels)

    # Подсчитываем количество каждого класса
    # train_class_counts = Counter(train_labels)
    # val_class_counts = Counter(val_labels)
    #
    # Вычисляем процентное соотношение
    # train_percentages = {full_dataset.classes[label]: count / len(train_labels) * 100
    #                      for label, count in train_class_counts.items()}
    # val_percentages = {full_dataset.classes[label]: count / len(val_labels) * 100
    #                    for label, count in val_class_counts.items()}
    #
    # Вывод результатов
    # print("Train class percentages:")
    # for class_name, percentage in train_percentages.items():
    #     print(f"{class_name}: {percentage:.2f}%")
    # print()
    #
    # print("\nValidation class percentages:")
    # for class_name, percentage in val_percentages.items():
    #     print(f"{class_name}: {percentage:.2f}%")
    # print()


    # Подготовка датасетов для конкретного фолда
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion)
        val_loss, val_acc, val_f1 = validate_one_epoch(model=model, dataloader=val_loader, criterion=criterion)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]\n",
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F-Score: {train_f1:.4f}\n",
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F-Score: {val_f1:.4f}")

    # Сохраняем результаты фолда
    fold_results.append((train_acc_history, val_acc_history, model.state_dict()))

    plt.plot(range(num_epochs), train_acc_history, label='Train Accuracy')
    plt.plot(range(num_epochs), val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f"Fold {fold+1}")
    plt.show()


# Теперь нужно визуализировать среднюю точность по всем фолдам на каждой эпохе на обучающей выборке и на валидационной.



#====================================================================
# Выберем лучший фолд по последней точности на валидации, например
#====================================================================
# best_fold = None
# best_val_acc = -1
# for i, (_, val_acc_hist, _) in enumerate(fold_results):
#     if val_acc_hist[-1] > best_val_acc:
#         best_val_acc = val_acc_hist[-1]
#         best_fold = i
# print(f"Best fold: {best_fold+1} with validation acc: {best_val_acc:.4f}")
#
# best_model_state = fold_results[best_fold][2]

# #====================================================================
# # 10. Сохраняем модель
# #====================================================================
# model_save_path = "alphabet_classifier.pth"
# torch.save({
#     'model_state_dict': best_model_state,
#     'classes': full_dataset.classes
# }, model_save_path)
# print("Model saved at:", model_save_path)

# #====================================================================
# # 11. Визуализация точности для лучшего фолда
# #====================================================================
# train_acc_hist, val_acc_hist, _ = fold_results[best_fold]
# plt.figure(figsize=(8, 6))
# plt.plot(train_acc_hist, label='Train Accuracy')
# plt.plot(val_acc_hist, label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()

#
# #====================================================================
# # 12. Загрузка модели и использование
# #====================================================================
# checkpoint = torch.load(model_save_path, map_location=device)
# model = SimpleCNN(num_classes=len(checkpoint['classes'])).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# class_names = checkpoint['classes']
#
# # Пример использования: предсказание для одного изображения
# import cv2
#
# test_image_path = 'path_to_single_letter.jpg'
# img = Image.open(test_image_path).convert('RGB')
# img_tensor = val_transform(img).unsqueeze(0).to(device) # [C, H, W] -> [1, C, H, W]
#
# with torch.no_grad():
#     outputs = model(img_tensor)
#     _, predicted = torch.max(outputs, 1)
#     predicted_class = class_names[predicted.item()]
#     print(f"Predicted class: {predicted_class}")
