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

def _build_ascan_samples(reciever_position_xy: tuple[float, float], time_window: tuple[float, float], n_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds the A-scan samples to use in predictions.
    """
    x, y = reciever_position_xy
    x, y = torch.tensor(x), torch.tensor(y)
    x = x.broadcast_to((n_steps))
    y = y.broadcast_to((n_steps))
    t = torch.linspace(time_window[0], time_window[1], n_steps)

    return x, y, t

def predict_ascan(f: Callable, reciever_position_xy: tuple[float, float], time_window: tuple[float, float], n_steps: int, device: str, geometry: torch.Tensor = None):
    """
    Predicts an A-scan with the given function.

    Parameters
    ----------
    f : Callable
        Prediction function
    reciever_position_xy : tuple[float, float]
        position of the reciever (x, y) in meters
    time_window : tuple[float, float]
        time window to predict in seconds
    n_steps : int
        number of points to predict inside the time window
    device : str
        device to use for predictions
    geometry : torch.Tensor, optional
        the geometry to use as input to the network. If None, then it will not be fed to f. By default None

    Returns
    -------
    torch.Tensor
        the predicted A-scan at the reciever
    """

    x, y, t = _build_ascan_samples(reciever_position_xy, time_window, n_steps)

    x = x.to(device)
    y = y.to(device)
    t = t.to(device)

    preds: torch.Tensor
    if geometry is None:
        preds = f(x, y, t)
    else:
        mlp_inputs = torch.stack([x, y, t], dim=-1)
        preds = f(mlp_inputs, geometry)

    return preds.flatten().detach()
