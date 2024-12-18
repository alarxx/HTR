import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from ctc.classificator.cnns import FCNN

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
                print(ch, char_to_idx[ch])
                seq.append(char_to_idx[ch])
            else:
                print(ch, "passed")
                # Игнорируем неизвестные символы
                pass
        return seq # АБВ. -> [1, 2, 3]

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        img_path = os.path.join(self.img_dir, sample['name'])
        image = Image.open(img_path).convert('RGB')

        text = sample['description']

        if self.transform:
            image = self.transform(image)

        target = self.text_to_sequence(text)
        target_length = len(target)

        # Возвращаем (image, target, target_length, text)
        return image, torch.tensor(target, dtype=torch.long), torch.tensor(target_length, dtype=torch.long), text


# Пример простой трансформации: преобразуем в тензор, приведём к одному размеру по высоте
transform = transforms.Compose([
    transforms.Resize((64, )),  # Пример: фиксируем высоту = 32, ширину до 1280 (или динамически)
    transforms.ToTensor()
])

dataset = HandwritingDataset(
    annotation_file="./KOHTD_dataset/HK_dataset/light_annotation.json",
    img_dir="./KOHTD_dataset/HK_dataset/img",
    transform=transform
)

# Проверим один образец
image, target, target_length, description = dataset[0]
print("Description (label):", description)
print("Target (int seq):", target)

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


dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

for batch_idx, (images, targets, target_lengths, texts) in enumerate(dataloader):
    print("batch_idx", batch_idx)
    print("images", images)
    print("targets", targets) # targets tensor([24, 14,  2, 24, 38, 22, 20, 23, 37, 18, 10, 28, 13,  8, 24, 38, 18,  7, 8,  5, 38])
    print("target_lengths", target_lengths) # target_lengths tensor([ 0,  1,  9, 11])
    print("texts", texts) # texts ('1', '1С:', 'Кәсіпорын', 'жүйесіндегі')


#############################################################################################################################################






