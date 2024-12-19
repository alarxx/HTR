import time

import torch
import torch.nn as nn
import torch.nn.functional as F


#############################################################
# Модель: Простая CNN + Linear для признаков
#
# Идея:
# 1) CNN уменьшает высоту до 1 (или небольшой размер),
# 2) На выходе получается тензор B x C x H' x W'
# 3) Превращаем в форму (W', B, C) и через линейный слой
#    получаем логиты классов для каждого "временного шага" (W').
#
# CNN Sliding Window:
# За счет вертикальному GMP CNN без явного цикла "скользящего окна", сама CNN создает
# "оконную" (последовательную) структуру данных по горизонтали.
#############################################################

class FCNN(nn.Module):
    def __init__(self, num_classes):
        super(FCNN, self).__init__()
        # Пример весьма простой архитектуры
        self.features = nn.Sequential(
            # ---1 convolution block---
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # Входные RGB-каналы -> 32 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25), # Dropout, p - probability of an element to be zeroed
            # ---2 convolution block---
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64 -> 64 фильтра
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # (64 -> 32)
            # nn.Dropout(0.25),                               # Dropout

            # ---3 convolution block---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 -> 128 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---4 convolution block---
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 -> 128 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # (32 -> 16)
            # nn.Dropout(0.25), # Dropout

            # ---5 сверточный блок---
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 -> 512 фильтров
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---6 convolution block---
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 512 -> 512 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # (16 -> 8)
            # nn.Dropout(0.25),

            # ---7 convolution block---
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---8 convolution block---
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),

            # 8 x ?
            nn.AdaptiveAvgPool2d((1, None))  # высоту до 1, ширина без изменения
        )
        # self.fc = nn.Linear(512, num_classes)  # Преобразуем фичи в логиты символов
        # Multilayer Perceptron
        self.fc = nn.Sequential(
            # ---9 linear block---
            nn.Linear(512, 256), # Полносвязный слой
            nn.LeakyReLU(),
            nn.Dropout(p=0.3), # Dropout для регуляризации
            # ---10 linear block---
            nn.Linear(256, num_classes) # Выходной слой с количеством классов
        )

    def forward(self, x):
        # print(x.shape)
        
        # x: B x 3 x H x W
        x = self.features(x)  # B x 512 x 1 x W'
        # Удаляем высоту (теперь = 1)
        x = x.squeeze(2)  # B x 512 x W'
        x = x.permute(2, 0, 1)  # W' x B x 512 (потому что CTC ждёт (T,B,C))
        logits = self.fc(x)     # W' x B x num_classes
        return logits



class FCNN_old(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Пример весьма простой архитектуры
        self.features = nn.Sequential(
            # ---1 convolution block---
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # Входные RGB-каналы -> 32 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25), # Dropout, p - probability of an element to be zeroed
            # ---2 convolution block---
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64 -> 64 фильтра
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # (64 -> 32)
            # nn.Dropout(0.25),                               # Dropout

            # ---3 convolution block---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 -> 128 фильтра
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---4 convolution block---
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 -> 128 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # (32 -> 16)
            # nn.Dropout(0.25), # Dropout

            # ---5 сверточный блок---
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 -> 512 фильтров
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---6 convolution block---
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 512 -> 512 фильтров
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # (16 -> 8)
            # nn.Dropout(0.25),

            # ---7 convolution block---
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.Dropout(0.25),
            # ---8 convolution block---
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),

            # 8 x ?
            nn.AdaptiveAvgPool2d((1, None))  # высоту до 1, ширина без изменения
        )
        self.fc = nn.Linear(512, num_classes)  # Преобразуем фичи в логиты символов

    def forward(self, x):
        # x: B x 3 x H x W
        x = self.features(x)  # B x 512 x 1 x W'
        # Удаляем высоту (теперь = 1)
        x = x.squeeze(2)  # B x 512 x W'
        x = x.permute(2, 0, 1)  # W' x B x 512 (потому что CTC ждёт (T,B,C))
        logits = self.fc(x)     # W' x B x num_classes
        return logits




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
            nn.Dropout(p=0.3),

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

