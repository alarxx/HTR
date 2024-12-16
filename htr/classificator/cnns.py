 
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyCNN12(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # Sequential stack of layers
        self.features = nn.Sequential(
            # 1 сверточный блок
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),     # Входные RGB-каналы -> 32 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25),                               # Dropout, p - probability of an element to be zeroed
            # 2 сверточный блок
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # 64 -> 64 фильтра
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                # (64 -> 32)
            # nn.Dropout(0.25),                               # Dropout

            # 3 сверточный блок
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # 64 -> 128 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # 4 сверточный блок
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 -> 128 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                # (32 -> 16)
            # nn.Dropout(0.25),                               # Dropout

            # 5 сверточный блок
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256 -> 512 фильтров
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # 6 сверточный блок
            # nn.Conv2d(256, 256, kernel_size=3, padding=1), # 512 -> 512 фильтров
            # nn.LeakyReLU(),
            nn.MaxPool2d(2),                               # (16 -> 8)
            # nn.Dropout(0.25),

            # 7 сверточный блок
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # 8 сверточный блок
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.25),

            # Global MaxPooling
            nn.AdaptiveMaxPool2d((1, 1))  # Преобразуем выходной тензор к размеру (1, 1)
        )

        # Полносвязный слой
        self.classifier = nn.Sequential(
            nn.Flatten(),               # "Сглаживаем" выходы сверточной сети
            nn.Linear(512, 256),        # Полносвязный слой
            nn.LeakyReLU(),
            nn.Dropout(0.5),            # Dropout для регуляризации
            nn.Linear(256, num_classes) # Выходной слой с количеством классов
        )

        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.features(x)         # CNN
        x = self.classifier(x)       # MLP
        return x # CrossEntropyLoss
        # return self.log_softmax(x) # NLLLoss


# Проверка сети
if __name__ == "__main__":
    model = FullyCNN12(num_classes=10)  # Например, 10 классов
    print(model)

    # Тестовые данные: 8 изображений размером 64x64 с 3 каналами (RGB)
    inputs = torch.randn(8, 3, 64, 64)
    outputs = model(inputs)
    print("Output shape:", outputs.shape)  # Ожидаемый размер: [8, 10]

    # Вычисление вероятностей из logits
    probabilities = F.softmax(outputs, dim=1)
    predicted_classes = probabilities.argmax(dim=1)

    print("probabilities shape:", probabilities.shape)  # Ожидаемый размер: [8, 10]
    print("probabilities:", probabilities[0])
    print("predicted_classes:", predicted_classes[0])
