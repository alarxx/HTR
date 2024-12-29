import torch
import matplotlib.pyplot as plt
from classificator.cnns import FCNN, FCNN_old  # Предполагается, что этот класс объявлен так же, как в training.py

# Loading the last model
checkpoint_path = "./FCNN_CTC_e35.pth"
checkpoint = torch.load(checkpoint_path)
train_loss_history = checkpoint['train_loss_history']
train_acc_history = checkpoint['train_acc_history']
val_acc_history = checkpoint['val_acc_history']

alphabet = [
    '<blank>', 'А', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е', 'Ё', 'Ж', 'З',
    'И', 'Й', 'К', 'Қ', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө', 'П', 'Р', 'С',
    'Т', 'У', 'Ұ', 'Ү', 'Ф', 'Х', 'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы',
    'І', 'Ь', 'Э', 'Ю', 'Я'
]
model = FCNN(num_classes=len(alphabet))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Levenshtein Accuracy (Train & Val)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train Levenshtein Ratio', marker='o')
plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label='Validation Levenshtein Ratio', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Levenshtein Ratio')
plt.title('Train and Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
