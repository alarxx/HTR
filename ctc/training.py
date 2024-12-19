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

from classificator.cnns import FCNN, FullyCNN10
from data_transforms.trans import MinMaxWidth, AddRandomNoise
from utils.convert import ntensor2cvmat


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


def levenshtein_ratio(preds, gts):
    total_ratio = 0
    for pred, gt in zip(preds, gts):
        # print(pred, gt)
        total_ratio += Levenshtein.ratio(pred, gt)
    return total_ratio / len(preds)  # Среднее значение Ratio


def greedy_decode(log_probs):
    # log_probs: (T, B, C)
    # Возьмём argmax по C
    argmaxes = torch.argmax(log_probs, dim=2)  # (T, B)
    # print(argmaxes)
    results = []
    for b in range(argmaxes.size(1)):
        seq = argmaxes[:, b].cpu().numpy()
        # Удалить повторяющиеся символы и blank-и
        decoded = []
        prev = None
        for s in seq:
            if s != 0 and s != prev:
                decoded.append(s)
            prev = s
        # Преобразуем индексы обратно в символы
        text = ''.join(idx_to_char[idx] for idx in decoded)
        results.append(text)
    return results


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
        image, target, target_length, text = self.subset[idx]
        # print("__getitem__ called:", img, label)
        if self.transform:
            image = self.transform(image)
        return image, target, target_length, text # torch.Tensor values in [0, 1], torch.float32, torch.Size([C, H, W])


#############################################################
# Алфавит и маппинги
#
# Добавляем символ blank под индексом 0, далее идут все буквы
#############################################################
# class_mapping = { # Пока не используется
#     "00": "А", "01": "Ә", "02": "Б", "03": "В", "04": "Г", "05": "Ғ",
#     "06": "Д", "07": "Е", "08": "Ё", "09": "Ж", "10": "З", "11": "И",
#     "12": "Й", "13": "К", "14": "Қ", "15": "Л", "16": "М", "17": "Н",
#     "18": "Ң", "19": "О", "20": "Ө", "21": "П", "22": "Р", "23": "С",
#     "24": "Т", "25": "У", "26": "Ұ", "27": "Ү", "28": "Ф", "29": "Х",
#     "30": "Һ", "31": "Ц", "32": "Ч", "33": "Ш", "34": "Щ", "35": "Ъ",
#     "36": "Ы", "37": "І", "38": "Ь", "39": "Э", "40": "Ю", "41": "Я"
# }
alphabet = [
    '<blank>', 'А', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е', 'Ё', 'Ж', 'З',
    'И', 'Й', 'К', 'Қ', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө', 'П', 'Р', 'С',
    'Т', 'У', 'Ұ', 'Ү', 'Ф', 'Х', 'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы',
    'І', 'Ь', 'Э', 'Ю', 'Я'
]
char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}


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
                # print(ch, char_to_idx[ch])
                seq.append(char_to_idx[ch])
            else:
                # print(ch, "passed")
                # Игнорируем неизвестные символы
                pass
        return seq  # АБВ. -> [1, 2, 3]

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        img_path = os.path.join(self.img_dir, sample['name'])
        image = Image.open(img_path).convert('RGB')

        # Получаем ширину и высоту изображения
        # width, height = image.size
        # print(f"Width: {width}, Height: {height}")  # Выводим ширину и высоту для отладки

        text = sample['description'].upper()

        if self.transform:
            image = self.transform(image)

        target = self.text_to_sequence(text)
        target_length = len(target)

        # image, target, target_length, text
        return image, torch.tensor(target, dtype=torch.long), torch.tensor(target_length, dtype=torch.long), text


#############################################################
# Collate function for DataLoader
#############################################################
def collate_fn(batch):
    # batch - список из элементов, возвращаемых __getitem__ в Dataset
    # Каждый элемент - это кортеж (image, target, target_length, text)
    images, targets, target_lengths, texts = zip(*batch)

    # Сначала паддим изображения (если нужно) до одной ширины внутри батча
    # 1. Находим максимальную высоту и ширину изображений в текущем батче
    max_width = max(img.shape[2] for img in images)
    max_height = max(img.shape[1] for img in images)

    # 2. Паддинг изображений до одного размера
    padded_images = []
    for img in images:
        _, h, w = img.shape # Получаем (C, H, W)
        # Создаем пустой (нулевой) тензор с размерами (C, max_height, max_width)
        padded = torch.zeros((3, max_height, max_width)) # 3 канала (RGB)
        # Копируем оригинальное изображение в этот тензор
        padded[:, :h, :w] = img
        padded_images.append(padded)

    # Складываем все изображения в один тензор (B x C x max_height x max_width)
    images = torch.stack(padded_images, dim=0)

    # 3. Конкатенируем метки
    # Вытаскиваем target из каждого элемента и объединяем в один длинный вектор
    targets = torch.cat(targets, dim=0) # ([1, 2, 3], [4, 5, 6]) -> [1, 2, 3, 4, 5, 6]
    # Преобразуем target_lengths в тензор
    target_lengths = torch.stack(target_lengths, dim=0) # [5, 3, 4]

    return images, targets, target_lengths, texts



#====================================================================
# Hyperparameters
#====================================================================
batch_size = 64
num_epochs = 1000
learning_rate = 0.0001
model_prefix="FCNN_CTC"


# Пример простой трансформации: преобразуем в тензор, приведём к одному размеру по высоте
transform = transforms.Compose([
    transforms.Resize((64, 256)), 
    transforms.ToTensor(),
    # MinMaxWidth(),
    AddRandomNoise(),
])
# Пример простой трансформации: преобразуем в тензор, приведём к одному размеру по высоте
train_transform = transforms.Compose([
    transforms.Resize((64, 256)),  
     # Геометрические трансформации
    transforms.RandomRotation(degrees=5),  # Поворот на +/-5 градусов
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=5),  
    # Можно поиграться с параметрами выше, чтобы не слишком сильно искажать текст
    # Перспективные искажения
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    # MinMaxWidth(),
    AddRandomNoise(amount=0.1),
])

dataset = HandwritingDataset(
    # annotation_file="./KOHTD_dataset/HK_dataset/merged_annotation.json",
    annotation_file="./KOHTD_dataset/HK_dataset/merged_annotation.json",
    img_dir="./KOHTD_dataset/HK_dataset/img",
)

train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

train_ds = CustomDataset(Subset(dataset, train_indices), transform=train_transform)
val_ds = CustomDataset(Subset(dataset, val_indices), transform=transform)
test_ds = CustomDataset(Subset(dataset, test_indices), transform=transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


for i in range(10):
    # Проверим один образец
    image, target, target_length, description = train_ds[i]
    print("Description (label):", description)
    print("Target (int seq):", target)
    print("target_length (?):", target_length)
    # Отображаем изображение
    plt.imshow(ntensor2cvmat(image))
    plt.title(f"{description}")
    plt.axis('off')
    plt.show()



for batch_idx, (images, targets, target_lengths, texts) in enumerate(train_loader):
    if batch_idx > 1:
        break
    print("batch_idx", batch_idx)
    print("images", images)
    print("targets", targets) # targets tensor([24, 14,  2, 24, 38, 22, 20, 23, 37, 18, 10, 28, 13,  8, 24, 38, 18,  7, 8,  5, 38])
    print("target_lengths", target_lengths) # target_lengths tensor([ 0,  1,  9, 11])
    print("texts", texts) # texts ('1', '1С:', 'Кәсіпорын', 'жүйесіндегі')

###################################################################################################################


#====================================================================
# Model
#
# CNN Sliding Window:
# За счет вертикальному GMP CNN без явного цикла "скользящего окна", сама CNN создает "оконную" (последовательную) структуру данных по горизонтали.
#====================================================================
checkpoint = torch.load('Alphabet_FCNN_10_4.pth', map_location=device)
pretrained_model = FullyCNN10(num_classes=len(checkpoint['classes'])).to(device)
pretrained_model.load_state_dict(checkpoint['model_state_dict'])
# Создаем новую модель FCNN
model = FCNN(num_classes=len(alphabet)).to(device)
# Извлекаем первые слои
pretrained_layers = list(pretrained_model.features.children())[:10]  # first four layer + maxpool
# Заменяем первые слои
model.features = nn.Sequential(
    *pretrained_layers,
    *list(model.features.children())[10:]  # Оставшиеся слои FCNN
)
# Проверяем, что замена прошла успешно
print(model)

# model = FCNN(num_classes=len(alphabet)).to(device)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#############################################################
# Функция обучения за одну эпоху
#############################################################
def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    batch_counter = 1

    for batch_idx, (images, targets, target_lengths, texts) in enumerate(dataloader):
        # print(texts)
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(images)
        # Для CTC нужен input_lengths
        # T = длина последовательности = W' (ширина фичей после CNN)
        # Предположим, что длина выхода для каждого в батче одинакова:
        T = logits.size(0)
        B = logits.size(1)
        # print(f"logits: (T, B, C): {logits.size()}")
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)

        # CTC Loss: (log_probs, targets, input_lengths, target_lengths)
        log_probs = F.log_softmax(logits, dim=2)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"batch_idx ({batch_idx}): loss={loss.item()}")


        # if (batch_counter) % 1000 == 0:  
        #     train_acc = validate_model(model, train_loader)
        #     val_acc = validate_model(model, val_loader)
        #     print(f"Epoch {epoch + 1}, batch: {batch_counter}, Train Loss: {train_loss}, Train Levenshtein Ratio: {train_acc}, Validation Levenshtein Ratio: {val_acc}")
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'batch_counter': batch_counter,
        #     }, f"{model_prefix}_e{epoch}_b{batch_counter}.pth")
        #     print(f"Saved model at batch {batch_counter}")

        total_loss += loss.item()       

        batch_counter+=1

    return total_loss / len(dataloader)


def validate_model(model, dataloader):
    model.eval()
    acc = 0
    all_samples = []
    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths, texts) in enumerate(dataloader):
            images = images.to(device)
            logits = model(images)  # (T, B, C)
            # log_probs = F.log_softmax(logits, dim=2)
            preds = greedy_decode(logits)
            acc += levenshtein_ratio(preds, texts)
            all_samples.extend(zip(preds, texts))

    # for pred, gt in zip(preds, texts):
    #     if random.random() < 0.01:
    #         print(f"Pred: {pred}\nGT: {gt}\n---")
    # Randomly select samples from the entire dataset
    random_samples = random.sample(all_samples, min(len(all_samples), 20))
    for pred, text in random_samples:
        print(f"Prediction: {pred}\nGround Truth: {text}\n{'-'*40}")

    return acc / len(dataloader)


train_loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} started!")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
    train_acc = validate_model(model, train_loader)
    val_acc = validate_model(model, val_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Train Levenshtein Ratio: {train_acc}, Validation Levenshtein Ratio: {val_acc}")

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    # if (epoch+1) % 10 == 0:
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }, f"{model_prefix}_e{epoch}.pth")


torch.save({
    'model_state_dict': model.state_dict(),
    'train_loss_history': train_loss_history,
    'train_acc_history': train_acc_history,
    'val_acc_history': val_acc_history,
}, f"{model_prefix}_main.pth")

test_acc = validate_model(model, test_loader)
print(f"Final Test: {test_acc}")


# Visualize Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Visualize Train and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train Levenshtein Ratio', marker='o')
plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label='Validation Levenshtein Ratio', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Levenshtein Ratio')
plt.title('Train and Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
