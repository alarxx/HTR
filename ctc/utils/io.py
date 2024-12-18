import torch
import pickle

class Savior:
    def __init__(self):
        pass

    # Функция для сохранения моделей
    def save_models(self, fold_results, full_dataset, prefix='model_fold'):
        for i, fold_result in enumerate(fold_results):
            train_acc_history, train_f1_history, train_loss_history, val_acc_history, val_f1_history, val_loss_history, model_state_dict = fold_result
            torch.save({
                'model_state_dict': model_state_dict,
                'classes': full_dataset.classes,
                'train_acc_history': train_acc_history,
                'train_f1_history': train_f1_history,
                'train_loss_history': train_loss_history,
                'val_acc_history': val_acc_history,
                'val_f1_history': val_f1_history,
                'val_loss_history': val_loss_history,
            }, f"{prefix}_{i}.pth")

            print(f"Model for fold {i} saved as {prefix}_{i}.pth")

    # Функция для загрузки моделей
    def load_models(self, ModelClass, prefix='model_fold', num_folds=0, device='cpu'):
        models = []
        metadata = []

        for i in range(num_folds):
            checkpoint = torch.load(f"{prefix}_{i}.pth", map_location=device)  # Загружаем файл
            model_state_dict = checkpoint['model_state_dict']
            classes = checkpoint['classes']

            # Создаём пустую модель и загружаем state_dict
            model = ModelClass(num_classes=len(classes))
            model.load_state_dict(model_state_dict)
            model.eval()  # Модель в режим предсказаний
            models.append(model)

            # Сохраняем дополнительные данные
            metadata.append({
                'train_acc_history': checkpoint['train_acc_history'],
                'train_f1_history': checkpoint['train_f1_history'],
                'train_loss_history': checkpoint['train_loss_history'],
                'val_acc_history': checkpoint['val_acc_history'],
                'val_f1_history': checkpoint['val_f1_history'],
                'val_loss_history': checkpoint['val_loss_history'],
                'classes': classes
            })

            print(f"Model for fold {i} loaded successfully!")

        return models, metadata

    def saveP(self, obj, path='fold_results.pkl'):
        # Сохраняем fold_results в файл
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
            print(f"Fold results saved to {path}")

    def loadP(self, path='fold_results.pkl'):
        # Загружаем fold_results из файла
        with open(path, 'rb') as f:
            results = pickle.load(f)
            return results


