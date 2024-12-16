import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter


if __name__ == "__main__":
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
    # Load Dataset
    # Assumed structure:
    # root_dir/class1/*.jpg
    # root_dir/class2/*.png
    # ...
    # Every sub-directory is a distinct class (a letter of alphabet, in our case)
    #====================================================================
    root_dir = './dataset'

    # https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder
    # ImageFolder inherits from DatasetFolder
    # transform can take transforms, but in our case we don't need it, we'll transform later
    # if allow_empty=True raises error if some of the sub-directories is empty
    full_dataset = ImageFolder(root=root_dir, transform=None, allow_empty=False) # -> Dataset - List[Tuple[<PIL.Image>, int]]

    print("full_dataset:", full_dataset)
    print()
    # Output: Dataset ImageFolder: Number of datapoints: 14, Root location: ./dataset
    print("full_dataset[0]:", full_dataset[0])
    # Output: (<PIL.Image.Image image mode=RGB size=41x56 at 0x7EFC13DBFF40>, 0)
    print("Classes:", full_dataset.classes)
    print("Class to index mapping:", full_dataset.class_to_idx)
    print()

    #====================================================================
    # Cross-validation
    #====================================================================
    indices = np.arange(len(full_dataset)) # [0, len]
    labels = [label for _, label in full_dataset]
    print("indices", indices)
    print("labels", labels)
    print()

    num_folds = 2
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
        print(f"Fold {fold+1}/{num_folds}")
        print("val_idx, train_idx:", val_idx, train_idx)

        # Получаем лейблы классов вместо индексов
        train_labels = [labels[idx] for idx in train_idx]
        val_labels = [labels[idx] for idx in val_idx]
        print("val_labels, train_labels:", val_labels, train_labels)

         # Подсчитываем количество каждого класса
        train_class_counts = Counter(train_labels)
        val_class_counts = Counter(val_labels)

        # Вычисляем процентное соотношение
        train_percentages = {full_dataset.classes[label]: count / len(train_labels) * 100 for label, count in train_class_counts.items()}
        val_percentages = {full_dataset.classes[label]: count / len(val_labels) * 100 for label, count in val_class_counts.items()}

        # Выводим результат
        print("Train class percentages:", train_percentages)
        print("Validation class percentages:", val_percentages)
        print()

        # Подготовка датасетов для конкретного фолда
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        print("train_dataset:", train_subset)
        print("val_dataset:", val_subset)

        print(f"Train dataset size: {len(train_subset)}")
        print(f"Validation dataset size: {len(val_subset)}")
        print("-" * 30)

        # # Применяем трансформации
        # # Можно сделать так: при итерировании мы можем использовать transform внутри DataLoader.
        # # Но проще переопределить data в памяти или создать свой класс.
        #
        # # Для наглядности создадим простой класс-обертку:
        # class CustomDataset(Dataset):
        #     def __init__(self, subset, transform):
        #         self.subset = subset
        #         self.transform = transform
        #
        #     def __len__(self):
        #         return len(self.subset)
        #
        #     def __getitem__(self, idx):
        #         img, label = self.subset[idx]
        #         if self.transform:
        #             img = self.transform(img)
        #         return img, label
        #
        # train_ds = CustomDataset(train_dataset, transform=train_transform)
        # val_ds = CustomDataset(val_dataset, transform=val_transform)
        #
        # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        # for images, labels in dataloader:

        #
        # num_classes = len(full_dataset.classes)
        # model = SimpleCNN(num_classes=num_classes).to(device)
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #
        # train_acc_history = []
        # val_acc_history = []
        #
        # for epoch in range(num_epochs):
        #     train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        #     val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        #
        #     train_acc_history.append(train_acc)
        #     val_acc_history.append(val_acc)
        #
        #     print(f"Epoch [{epoch+1}/{num_epochs}] "
        #         f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
        #         f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        #
        # # Сохраняем результаты фолда
        # fold_results.append((train_acc_history, val_acc_history, model.state_dict()))


