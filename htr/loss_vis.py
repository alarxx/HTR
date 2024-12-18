import torch
import numpy as np
import matplotlib.pyplot as plt

# Define a helper function to load the models and extract train losses
def load_and_extract_losses(prefix, num_folds=5):
    """
    Load models and extract train loss histories.
    Args:
        prefix (str): Prefix for the model filenames.
        num_folds (int): Number of folds (models to load).
    Returns:
        list of lists: Train loss histories for all models.
    """
    train_loss_histories = []
    val_loss_histories = []
    for i in range(num_folds):
        checkpoint_path = f"{prefix}_{i}.pth"
        checkpoint = torch.load(checkpoint_path)
        train_loss_histories.append(checkpoint['train_loss_history'])
        val_loss_histories.append(checkpoint['val_loss_history'])
        print(f"Loaded train loss history from model {prefix}_{i}.pth")
    return train_loss_histories, val_loss_histories

# Load train losses for all 5 models
prefix = "Alphabet_FCNN_10"
num_folds = 5
train_loss_histories, val_loss_histories = load_and_extract_losses(prefix, num_folds)

# Average the train losses across all folds
num_epochs = len(train_loss_histories[0])  # Assuming all histories have the same length
avg_train_loss = np.mean(np.array(train_loss_histories), axis=0)
avg_val_loss = np.mean(np.array(val_loss_histories), axis=0)

# Plot the averaged train losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), avg_train_loss, label='Averaged Train Loss', marker='o')
plt.plot(range(1, num_epochs + 1), avg_val_loss, label='Averaged Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f"Averaged Losses Across {num_folds} Folds")
plt.legend()
plt.grid(True)
plt.show()
