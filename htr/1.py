from htr.classificator.cnns import FullyCNN10
from utils.io import Savior
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    savior = Savior()
    models, metadata = savior.load_models(ModelClass=FullyCNN10, prefix="Alphabet_FCNN", num_folds=2, device=device)
    print(models)
    print(metadata)