import torch
import torch.nn as nn
import torch.optim as optim

features = 8

# Параметры данных и модели
T = 10  # Длина входной последовательности
C = 4  # Число классов (включая blank). Пусть classes = {0: blank, 1, 2, 3}
B = 1  # Размер batch = 1 для наглядности
S = 4  # Длина целевой последовательности
S_min = S  # Минимальная длина = длина таргета


# Сгенерируем фиксированные входные данные, которые мы хотим "запомнить".
# Допустим, вход – это шум, но фиксированный, чтобы модель могла "запомнить" отображение.
fixed_input = torch.randn(T, B, features)  # 10 фич на каждый из T шагов, батч из 1

# Допустим, мы хотим, чтобы модель предсказывала последовательность [1, 2, 2, 1].
# Это наша «целевая» последовательность.
target_seq = torch.tensor([[1, 2, 2, 3]], dtype=torch.long)  # размер (B, S)
print("target_seq", target_seq)

# Чтобы CTCLoss работал, нам нужны длины входов и таргетов.
input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
print("input_lengths", input_lengths)

target_lengths = torch.full(size=(B,), fill_value=S, dtype=torch.long)
print("target_lengths", target_lengths)

# Создадим простейшую модель: Линейный слой, который будет выдавать логиты C классов для каждого временного шага.
# Мы будем подавать туда просто рандомный тензор (имитируя входные фичи, скажем, из RNN/ трансформера).
model = nn.Linear(features, C)  # допустим у нас 8 признаков на каждый временной шаг
logSoftmax = nn.LogSoftmax(dim=2)

# Функция ошибки CTC
ctc_loss = nn.CTCLoss(blank=0)

# Оптимизатор
optimizer = optim.SGD(model.parameters(), lr=0.1)


# Попытаемся обучать модель несколько итераций и посмотрим, как убывает loss.
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Прогон через модель: получаем логиты размера (T, B, C)
    logits = model(fixed_input)  # (T, B, C)
    log_probs = logSoftmax(logits)  # (T, B, C), лог-пробабилити

    print("logits", logits)
    # print("log_probs", log_probs)

    # Считаем loss
    loss = ctc_loss(log_probs, target_seq, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}")
