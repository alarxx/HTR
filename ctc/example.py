import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from classificator.cnns import FCNN, FullyCNN10
from data_transforms.trans import MinMaxWidth, AddRandomNoise
from utils.convert import ntensor2cvmat

# class_mapping = { # Пока не используется
#     "00": "А", "01": "Ә", "02": "Б", "03": "В", "04": "Г", "05": "Ғ",
#     "06": "Д", "07": "Е", "08": "Ё", "09": "Ж", "10": "З", "11": "И",
#     "12": "Й", "13": "К", "14": "Қ", "15": "Л", "16": "М", "17": "Н",
#     "18": "Ң", "19": "О", "20": "Ө", "21": "П", "22": "Р", "23": "С",
#     "24": "Т", "25": "У", "26": "Ұ", "27": "Ү", "28": "Ф", "29": "Х",
#     "30": "Һ", "31": "Ц", "32": "Ч", "33": "Ш", "34": "Щ", "35": "Ъ",
#     "36": "Ы", "37": "І", "38": "Ь", "39": "Э", "40": "Ю", "41": "Я"
# }

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

        text = sample['description'].upper()

        if self.transform:
            image = self.transform(image)

        target = self.text_to_sequence(text)
        target_length = len(target)

        # image, target, target_length, text
        return image, torch.tensor(target, dtype=torch.long), torch.tensor(target_length, dtype=torch.long), text

# Пример простой трансформации: преобразуем в тензор, приведём к одному размеру по высоте
transform = transforms.Compose([
    # transforms.Resize((32, 256)),  # Пример: фиксируем высоту = 32, ширину до 1280 (или динамически)
    transforms.ToTensor(),
    MinMaxWidth(),
    AddRandomNoise(),
])

dataset = HandwritingDataset(
    # annotation_file="./KOHTD_dataset/HK_dataset/merged_annotation.json",
    annotation_file="./KOHTD_dataset/HK_dataset/light_annotation.json",
    img_dir="./KOHTD_dataset/HK_dataset/img",
    transform=transform
)

# for i in range(4):
#     # Проверим один образец
#     image, target, target_length, description = dataset[i]
#     print("Description (label):", description)
#     print("Target (int seq):", target)
#     print("target_length (?):", target_length)
#     # Отображаем изображение
#     plt.imshow(ntensor2cvmat(image))
#     plt.title(f"{description}")
#     plt.axis('off')
#     plt.show()



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

batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

for batch_idx, (images, targets, target_lengths, texts) in enumerate(dataloader):
    if batch_idx > 1:
        break
    print("batch_idx", batch_idx)
    print("images", images)
    print("targets", targets) # targets tensor([24, 14,  2, 24, 38, 22, 20, 23, 37, 18, 10, 28, 13,  8, 24, 38, 18,  7, 8,  5, 38])
    print("target_lengths", target_lengths) # target_lengths tensor([ 0,  1,  9, 11])
    print("texts", texts) # texts ('1', '1С:', 'Кәсіпорын', 'жүйесіндегі')

###################################################################################################################

learning_rate = 0.0001

checkpoint = torch.load('Alphabet_FCNN_10_4.pth', map_location=device)
pretrained_model = FullyCNN10(num_classes=len(checkpoint['classes'])).to(device)
pretrained_model.load_state_dict(checkpoint['model_state_dict'])


# Создаем новую модель FCNN
model = FCNN(num_classes=len(alphabet)).to(device)

# Заменяем первые 5 слоев
# Извлекаем первые 5 слоев из pretrained_model
pretrained_layers = list(pretrained_model.features.children())[:9]  # first four

# Заменяем первые 5 слоев в features у FCNN
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
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0

    train_acc_history = []
    train_f1_history = []
    train_loss_history = []
    val_acc_history = []
    val_f1_history = []
    val_loss_history = []
    
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

        # print(f"batch_idx ({batch_idx}): loss={loss.item()}")

        total_loss += loss.item()

    return total_loss / len(dataloader)


epochs = 100

for epoch in range(epochs):
    print(f"Epoch {epoch+1} started!")
    loss = train_one_epoch(model, dataloader, criterion, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss}")

#############################################################
# CNN Sliding Window:
# За счет вертикальному GMP CNN без явного цикла "скользящего окна", сама CNN создает
# "оконную" (последовательную) структуру данных по горизонтали.
#############################################################

def greedy_decode(log_probs):
    # log_probs: (T, B, C)
    # Возьмём argmax по C
    argmaxes = torch.argmax(log_probs, dim=2)  # (T, B)
    print(argmaxes)
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

def test_model(model, dataloader):
    model.eval()
    total_samples = 0
    # Счётчик для примера; для метрик нужен отдельный подсчёт CER/WER
    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths, texts) in enumerate(dataloader):
            print(texts)
            images = images.to(device)
            logits = model(images)  # (T, B, C)
            # log_probs = F.log_softmax(logits, dim=2)
            preds = greedy_decode(logits)  # Список строк (предсказания)
            for pred, gt in zip(preds, texts):
                print(f"Pred: {pred}\nGT: {gt}\n---")

test_model(model, dataloader)
