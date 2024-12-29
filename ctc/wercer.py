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

from classificator.cnns import FCNN, FCNN_old, FullyCNN10
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
# Возвращаем также среднюю уверенность
#############################################################
def greedy_decode(logits):
    # logits: [T, B, C]
    probs = F.softmax(logits, dim=2)  # [T, B, C]
    argmaxes = torch.argmax(probs, dim=2) # [T, B]
    results = []
    confidences = []
    for b in range(argmaxes.size(1)):
        seq = argmaxes[:, b].cpu().numpy()
        decoded = []
        char_confidences = []
        prev = None
        for t, s in enumerate(seq):
            if s != 0 and s != prev:
                decoded.append(s)
                char_confidences.append(probs[t, b, s].item())
            prev = s
        # text = ''.join(idx_to_char[idx] for idx in decoded)
        text = ''.join(idx_to_char[idx] for idx in decoded if idx in idx_to_char)
        results.append(text)

        if len(char_confidences) > 0:
            avg_conf = sum(char_confidences) / len(char_confidences)
        else:
            avg_conf = 0.0
        confidences.append(avg_conf)

    return results, confidences

#############################################################
# Загрузка обученной модели
#############################################################
model_prefix = "FCNN_CTC"
checkpoint_path = f"./{model_prefix}_e35.pth"

model = FCNN(num_classes=len(alphabet)).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

#############################################################
# Подсчёт WER и CER на тестовом датасете
#############################################################
total_char_distance = 0
total_chars = 0
total_word_distance = 0
total_words = 0

with torch.no_grad():
    for images, targets, target_lengths, texts in test_loader:
        images = images.to(device)
        logits = model(images)  # [T, B, C], предположим T это ширина фичей, B=batch
        # Для CTC декодирования предполагается (T, B, C)
        # Если модель возвращает (B, C, W, ...), надо преобразовать. Предположим, что logits уже в нужной форме.
        # Если нет, то надо дополнительно транспонировать:
        if logits.size(0) != images.size(0):
            # Предположим, что модель возвращает (B, C, W), надо поменять на (W, B, C)
            logits = logits.permute(2, 0, 1)  # если форма была (B, C, W)

        preds, confs = greedy_decode(logits)

        # Подсчёт ошибок
        for pred, gt in zip(preds, texts):
            # CER
            char_dist = Levenshtein.distance(pred, gt)
            total_char_distance += char_dist
            total_chars += len(gt)

            # WER (разбиваем по пробелам, если есть)
            pred_words = pred.split()
            gt_words = gt.split()
            word_dist = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words))
            total_word_distance += word_dist
            total_words += len(gt_words)

CER = total_char_distance / total_chars if total_chars > 0 else 0.0
WER = total_word_distance / total_words if total_words > 0 else 0.0

print("Test CER: {:.4f}".format(CER))
print("Test WER: {:.4f}".format(WER))
