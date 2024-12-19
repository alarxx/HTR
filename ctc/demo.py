import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import Levenshtein

from classificator.cnns import FCNN_old, FullyCNN10
from data_transforms.trans import MinMaxWidth, AddRandomNoise
from utils.convert import ntensor2cvmat

#====================================================================
# Определение устройства (CPU/GPU)
#====================================================================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("device:", device)
print()

#############################################################
# Алфавит и маппинги
#############################################################
alphabet = [
    '<blank>', 'А', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е', 'Ё', 'Ж', 'З',
    'И', 'Й', 'К', 'Қ', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө', 'П', 'Р', 'С',
    'Т', 'У', 'Ұ', 'Ү', 'Ф', 'Х', 'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы',
    'І', 'Ь', 'Э', 'Ю', 'Я'
]
char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}


#############################################################
# Класс датасета
#############################################################
class HandwritingDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def text_to_sequence(self, text):
        seq = []
        for ch in text:
            ch = ch.upper()
            if ch in char_to_idx:
                seq.append(char_to_idx[ch])
            else:
                # Игнорируем неизвестные символы
                pass
        return seq

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        img_path = os.path.join(self.img_dir, sample['name'])
        image = Image.open(img_path).convert('RGB')
        text = sample['description'].upper()

        if self.transform:
            image = self.transform(image)

        target = self.text_to_sequence(text)
        target_length = len(target)

        return image, torch.tensor(target, dtype=torch.long), torch.tensor(target_length, dtype=torch.long), text


#############################################################
# Обёртка для датасета с transforms
#############################################################
class CustomDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image, target, target_length, text = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, target, target_length, text

#############################################################
# Функция collate для DataLoader
#############################################################
def collate_fn(batch):
    images, targets, target_lengths, texts = zip(*batch)

    # Паддинг по размеру внутри батча
    max_width = max(img.shape[2] for img in images)
    max_height = max(img.shape[1] for img in images)

    padded_images = []
    for img in images:
        _, h, w = img.shape
        padded = torch.zeros((3, max_height, max_width))
        padded[:, :h, :w] = img
        padded_images.append(padded)

    images = torch.stack(padded_images, dim=0)

    targets = torch.cat(targets, dim=0)
    target_lengths = torch.stack(target_lengths, dim=0)

    return images, targets, target_lengths, texts

#############################################################
# Инициализация датасета и даталоадеров
#############################################################
transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
    AddRandomNoise(),
])

train_transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    AddRandomNoise(amount=0.1),
])

dataset = HandwritingDataset(
    annotation_file="./KOHTD_dataset/HK_dataset/merged_annotation.json",
    img_dir="./KOHTD_dataset/HK_dataset/img",
)

train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

train_ds = CustomDataset(Subset(dataset, train_indices), transform=train_transform)
val_ds = CustomDataset(Subset(dataset, val_indices), transform=transform)
test_ds = CustomDataset(Subset(dataset, test_indices), transform=transform)

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#############################################################
# Функция для декодирования выхода модели (greedy decode)
#############################################################
def greedy_decode(logits):
    argmaxes = torch.argmax(logits, dim=2)
    results = []
    for b in range(argmaxes.size(1)):
        seq = argmaxes[:, b].cpu().numpy()
        decoded = []
        prev = None
        for s in seq:
            if s != 0 and s != prev:
                decoded.append(s)
            prev = s
        text = ''.join(idx_to_char[idx] for idx in decoded)
        results.append(text)
    return results

#############################################################
# Загрузка обученной модели
# Предполагается, что чекпоинт "FCNN_CTC_main.pth" существует
#############################################################
model_prefix = "FCNN_CTC"
checkpoint_path = f"{model_prefix}_e14.pth"

# Создаем модель
model = FCNN_old(num_classes=len(alphabet)).to(device)

# Загружаем веса
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

#############################################################
# Демонстрация результата на случайных образцах из ds
#############################################################

ds = test_ds
ds = val_ds

num_samples = 3  # Кол-во случайных примеров

# Настройка подграфиков для вывода изображений
fig, axes = plt.subplots(nrows=num_samples, ncols=1, figsize=(8, num_samples*2))
if num_samples == 1:
    axes = [axes]

for i in range(num_samples):
    idx = random.randint(0, len(ds)-1)
    image, target, target_length, text = ds[idx]

    # Подготовка изображения для модели
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_input)
        preds = greedy_decode(logits)
        recognized = preds[0] if len(preds) > 0 else ""

    # Вывод результата
    print(f"{text} -> {recognized}")

    # Отображение изображения
    img_show = ntensor2cvmat(image)
    axes[i].imshow(img_show)
    axes[i].set_title(f"GT: {text}\nPredicted: {recognized}", fontproperties=None)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
