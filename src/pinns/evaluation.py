from typing import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def predict_loader(predict_f: Callable, loader: DataLoader, device: str):
    predictions = []
    with torch.no_grad:
        for batch in loader:
            
            for data in batch:
                data = data.to(device)
            
            # everything is an input except the last value, which is the ground truth
            pred = predict_f(*batch[:-1])
            pred = pred.cpu()
            predictions.append(pred)
    predictions = torch.cat(predictions)
    return predictions

def get_ground_truth(loader: DataLoader):
    gt = []
    for batch in loader:
        gt.append(batch[-1])
    
    return torch.cat(gt)

from src.visualization.misc import save_field_animation

def show_predictions_loader(f_PINN: Callable, f_regular: Callable, loader: torch.Tensor, device: str, img_size : tuple, save_path: str | Path = None):
    # show predictions of the field for NN and PINN
    ground_truth = get_ground_truth(loader)
    ground_truth = ground_truth.reshape((-1, *img_size))
    regular_predictions =  predict_loader(f_regular, loader, device)
    regular_predictions = regular_predictions.reshape((-1, *img_size))
    PINN_predictions =  predict_loader(f_PINN, loader, device)
    PINN_predictions = PINN_predictions.reshape((-1, *img_size))

    save_path = Path(save_path)
    save_field_animation(ground_truth, save_path / "ground truth")
    save_field_animation(PINN_predictions, save_path / "PINN predictions")
    save_field_animation(regular_predictions, save_path / "NN predictions")


