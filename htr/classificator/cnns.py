import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyCNN10(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # Sequential stack of convolutional layers
        self.features = nn.Sequential(
            # ---1 convolution block---
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), # Входные RGB-каналы -> 32 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25), # Dropout, p - probability of an element to be zeroed
            # ---2 convolution block---
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64 -> 64 фильтра
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                # (64 -> 32)
            # nn.Dropout(0.25),                               # Dropout

            # ---3 convolution block---
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 -> 128 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---4 convolution block---
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 128 -> 128 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # (32 -> 16)
            # nn.Dropout(0.25), # Dropout

            # ---5 сверточный блок---
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256 -> 512 фильтров
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---6 convolution block---
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 512 -> 512 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # (16 -> 8)
            # nn.Dropout(0.25),

            # ---7 convolution block---
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---8 convolution block---
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.25),

            # ---Global MaxPooling---
            nn.AdaptiveMaxPool2d((1, 1))  # Преобразуем выходной тензор к размеру (1, 1)
        )

        # Multilayer Perceptron
        self.classifier = nn.Sequential(
            nn.Flatten(), # выходы сверточной сети
            # ---9 linear block---
            nn.Linear(512, 256), # Полносвязный слой
            nn.LeakyReLU(),
            nn.Dropout(p=0.3), # Dropout для регуляризации
            # ---10 linear block---
            nn.Linear(256, num_classes) # Выходной слой с количеством классов
        )

        # self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.features(x)         # CNN
        x = self.classifier(x)       # MLP
        return x # CrossEntropyLoss
        # return self.log_softmax(x) # NLLLoss


class FullyCNN12(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # Sequential stack of convolutional layers
        self.features = nn.Sequential(
            # ---1 convolution block---
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), # Входные RGB-каналы -> 32 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25), # Dropout, p - probability of an element to be zeroed
            # ---2 convolution block---
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64 -> 64 фильтра
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 -> 64 фильтра
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                # (64 -> 32)
            # nn.Dropout(0.25),                               # Dropout

            # ---3 convolution block---
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 64 -> 128 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---4 convolution block---
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 128 -> 128 фильтров
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 128 -> 128 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # (32 -> 16)
            # nn.Dropout(0.25), # Dropout

            # ---5 сверточный блок---
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256 -> 512 фильтров
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---6 convolution block---
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 512 -> 512 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # (16 -> 8)
            # nn.Dropout(0.25),

            # ---7 convolution block---
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---8 convolution block---
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.25),

            # ---Global MaxPooling---
            nn.AdaptiveMaxPool2d((1, 1))  # Преобразуем выходной тензор к размеру (1, 1)
        )

        # Multilayer Perceptron
        self.classifier = nn.Sequential(
            nn.Flatten(), # выходы сверточной сети
            # ---9 linear block---
            nn.Linear(512, 256), # Полносвязный слой
            nn.LeakyReLU(),
            nn.Dropout(p=0.3), # Dropout для регуляризации
            # ---10 linear block---
            nn.Linear(256, num_classes) # Выходной слой с количеством классов
        )

        # self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.features(x)         # CNN
        x = self.classifier(x)       # MLP
        return x # CrossEntropyLoss
        # return self.log_softmax(x) # NLLLoss



# Проверка сети
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = FullyCNN10(num_classes=10).to(device)  # Например, 10 классов
    print(model)

    start_time = time.time()
    num_iterations = 1000
    for _ in range(num_iterations):
        # Тестовые данные: 8 изображений размером 64x64 с 3 каналами (RGB)
        inputs = torch.randn(32, 3, 64, 64).to(device)
        outputs = model(inputs)

    end_time = time.time()
    print(f"Total time for {num_iterations} iterations: {end_time - start_time:.2f} seconds") # 8 secs

    inputs = torch.randn(1, 3, 64, 64).to(device)
    outputs = model(inputs)
    print("Output shape:", outputs.shape)  # Ожидаемый размер: [8, 10]

    # Вычисление вероятностей из logits
    probabilities = F.softmax(outputs, dim=1)
    predicted_classes = probabilities.argmax(dim=1)

    print("probabilities shape:", probabilities.shape)  # Ожидаемый размер: [8, 10]
    print("probabilities:", probabilities[0])
    print("predicted_classes:", predicted_classes[0])
