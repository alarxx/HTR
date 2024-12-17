import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

#############################################################
# Алфавит и маппинги
#
# Добавляем символ blank под индексом 0, далее идут все буквы
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
# Датасет
#############################################################
class WordDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)  # список объектов {"name":..., "description":...}

    def __len__(self):
        return len(self.annotations)

    def text_to_sequence(self, text):
        seq = []
        for ch in text:
            if ch in char_to_idx:
                seq.append(char_to_idx[ch])
            else:
                # Если вдруг символ отсутствует в словаре, можно игнорировать или заменить
                # Игнорируем неизвестные символы
                pass
        return seq

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        img_path = sample["name"]
        text = sample["description"]

        img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        target = self.text_to_sequence(text)
        target_length = len(target)

        return img, torch.tensor(target, dtype=torch.long), torch.tensor(target_length, dtype=torch.long), text


#############################################################
# Коллектор для DataLoader, чтобы объединять в батч
#
# В батче могут быть последовательности разной длины.
# Для CTC мы просто склеиваем targets в один вектор (как того требует CTC),
# и отдельно храним длины каждого таргета.
#############################################################
def collate_fn(batch):
    # batch - список элементов (img, target, target_length, original_text)
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    target_lengths = [item[2] for item in batch]
    texts = [item[3] for item in batch]

    images = torch.stack(images, dim=0)  # (N, C, H, W)

    # Склеиваем все таргеты в один массив
    targets_cat = torch.cat(targets, dim=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets_cat, target_lengths, texts


#############################################################
# Модель (Fully CNN для CTC)
#
# Архитектура:
# - Свёртки и пулинг, чтобы уменьшить высоту до 1
# - На выходе Conv, маппящий в количество классов
# - Не схлопываем в один вектор, а сохраняем измерение ширины как "время"
#
# Выход: (T, N, C)
#############################################################
class FullyCNN12CTC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Упростим архитектуру, но оставим идею, схожую с Вашей:
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),  # Уменьшаем H и W в 2 раза

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),  # Еще в 2 раза

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),  # Еще в 2 раза

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # На данный момент высота уменьшена в 2*2*2=8 раз.
            # Если изначально H=32, то после 3 пулов H=32/8=4.
            # Чтобы получить высоту = 1, можно еще раз применить pooling только по высоте.
            # Но для простоты будем считать, что у нас уже H довольно маленькая.
            # Если хотим точно 1, можно сделать AdaptiveAvgPool2d((1, None))
        )

        # Сжимаем высоту до 1
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))  # Высота = 1, ширина не меняется

        # Преобразуем число каналов (512) в num_classes
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.features(x)  # (B, 512, H', W')
        features = self.height_pool(features)  # (B, 512, 1, W')

        # Применяем последний Conv
        logits = self.classifier(features)  # (B, num_classes, 1, W')

        # Для CTC: (T, N, C)
        # W' будет играть роль T (длина последовательности)
        # Переставляем оси: (B, C, 1, W') -> (W', B, C)
        logits = logits.squeeze(2)  # (B, num_classes, W')
        logits = logits.permute(2, 0, 1)  # (W', B, num_classes)

        return logits


#############################################################
# Функции декодирования (greedy)
# Применяются после того, как модель выдала логиты.
#############################################################
def greedy_decode(logits):
    # logits: (T, N, C)
    # Берем argmax по C
    pred_indices = torch.argmax(logits, dim=2)  # (T, N)
    # Удаляем последовательные повторы и blank
    results = []
    for n in range(pred_indices.size(1)):
        seq = pred_indices[:, n].cpu().numpy()
        # Применяем CTC greedy decoding
        decoded = []
        prev = None
        for ch_idx in seq:
            if ch_idx != 0 and ch_idx != prev:
                decoded.append(ch_idx)
            prev = ch_idx
        text = ''.join([idx_to_char[i] for i in decoded])
        results.append(text)
    return results


#############################################################
# Загрузка данных, разделение на train/val
#############################################################
transform = transforms.Compose([
    transforms.Resize((64, 160)),  # фиксированный размер для упрощения
    transforms.ToTensor(),
])

dataset = WordDataset(root_dir='./KOHTD_dataset/HK_dataset/img', annotations_file='./KOHTD_dataset/HK_dataset/merged_annotation.json', transform=transform)

train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(dataset, train_indices)
val_subset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, collate_fn=collate_fn)

#############################################################
# Обучение
#############################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(alphabet)

model = FullyCNN12CTC(num_classes=num_classes).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, targets, target_lengths, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        # Прогон через модель
        logits = model(images)  # (T, N, C)
        T = logits.size(0)
        N = logits.size(1)

        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)

        loss = criterion(logits, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets, target_lengths, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            T = logits.size(0)
            N = logits.size(1)
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(5):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#############################################################
# Пример предсказания
#############################################################
model.eval()
with torch.no_grad():
    for images, targets, target_lengths, original_texts in val_loader:
        images = images.to(device)
        logits = model(images)  # (T, N, C)
        pred_texts = greedy_decode(logits)
        for ot, pt in zip(original_texts, pred_texts):
            print(f"Original: {ot} | Predicted: {pt}")
        break  # показать пример из первого батча
