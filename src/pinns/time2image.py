"""
This module contains the application of CNN based PINNs to the railway dataset
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from kornia.filters import spatial_gradient, gaussian_blur2d
from skimage.measure import block_reduce

from src.visualization.field import save_field_animation
from src.pinns.models import Time2Image

import matplotlib.pyplot as plt

PINN = False
DEVICE = "cuda:2"
WARMUP_EPOCHS = 10000
EPOCHS = 3000
RNG = torch.quasirandom.SobolEngine(1, False)

collocation_points = RNG.draw(50).to(DEVICE)
collocation_points = collocation_points * (10e-9) + 2e-9
collocation_points.requires_grad_()

EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6

def is_close(t1: torch.Tensor, t2: torch.Tensor, tolerance: float = 0.001):
    """
    Checks if the given tensors are close one another

    Parameters
    ----------
    t1 : torch.Tensor
        first tensor
    t2 : torch.Tensor
        second tensor
    tolerance : float
        max relative difference accepted
    
    Returns
    -------
    bool
        True if the tensors are close, false otherwise
    """
    diff = (t1 - t2) / (t1.abs() + t2.abs())

    plt.imshow(diff.cpu().detach())
    plt.show()
    print(diff.abs().max())
    return torch.all(diff.abs() > tolerance).item()
    


def PINN_loss_fn(f: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor, labels: torch.Tensor, loss_fn: Callable, n_collocations: int, coll_domain: tuple[float, float], geometry: torch.Tensor):

    # Observation loss:
    preds = f(t)
    observation_loss = loss_fn(preds, labels)

    # how to compute gradient w.r.t output image?
    # use torch.func.jvp: it is forward automatic differentiation. 1 input -> multiple outputs
    def f_grad_aux(t):
        """
        Auxilliary function to compute first and second order gradients in 1 go
        """
        preds, dft = torch.func.jvp(f, (t,), (torch.ones_like(t),))
        return dft, preds
    
    dft, dftt, coll_preds = torch.func.jvp(f_grad_aux, (collocation_points,), (torch.ones_like(collocation_points),), has_aux=True)
    spatial_gradients2 = spatial_gradient(coll_preds.unsqueeze(1), mode="sobel", order=2, normalized=True)
    # scale the spatial gradients so that they express the real measurements in meters.
    # 1 cell is 6mm -> 2nd order gradients need to be multiplied by 1/(0.006)**2
    spatial_gradients2 = spatial_gradients2 * (1/(0.006)**2)
    dfxx = spatial_gradients2[:, :, 0, :, :].squeeze()
    # dfxx = gaussian_blur2d(dfxx, (3, 3), (1., 1.))
    dfyy = spatial_gradients2[:, :, 1, :, :].squeeze()
    # dfyy = gaussian_blur2d(dfyy, (3, 3), (1., 1.))

    # cut the images
    dft = dft[:, 2:-2, 2:-2]
    dftt = dftt[:, 2:-2, 2:-2]
    dfxx = dfxx[:, 2:-2, 2:-2]
    dfyy = dfyy[:, 2:-2, 2:-2]
    geometry = geometry[:, 2:-2, 2:-2]

    epsilon, sigma, mu, _ = geometry

    epsilon = epsilon * EPSILON_0
    mu = mu * MU_0

    # print(dftt.shape)
    # save_field_animation(dftt.cpu().detach(), None, 1)


    term1 = dftt
    term2 = -(1/(epsilon*mu)) * (dfxx + dfyy)
    term3 = dft * sigma / epsilon
    # term3 = 0

    collocation_loss = term1 + term2 + term3
    collocation_loss = loss_fn(1e-18 * collocation_loss, torch.zeros_like(collocation_loss))
    print(collocation_loss)
    physics_loss = collocation_loss

    return observation_loss, physics_loss


def get_f(model: Time2Image, input_scale: float, output_scale: float):
    input_scale = torch.tensor(input_scale)
    output_scale = torch.tensor(output_scale)
    def f(t):
        t = t * input_scale
        out= model(t)
        out = out.squeeze() * output_scale
        out = out[:, 2:-2, 3:-3]
        return out
    return f

def predict_all(f, times):
    with torch.no_grad():
        p = f(times.unsqueeze(-1))
    return p.cpu()

def time2Image(black_box: bool):

    geometry = np.load("munnezza/output/scan_00000/scan_00000_geometry.npy")
    snapshots = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_E"]

    geometry = block_reduce(geometry, block_size=(1, 3, 3), func=np.mean)
    geometry = torch.from_numpy(geometry).to(DEVICE)

    # snapshots = np.load("paper_data/2layer_wavefield.npz")["00000_E"]
    #snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]

    snapshots = torch.from_numpy(snapshots).to(DEVICE, dtype=torch.float32)
    #times = np.load("paper_data/uniform_wavefield.npz")["0000_times"]
    times = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_times"]
    times = torch.from_numpy(times).to(DEVICE, dtype=torch.float32)

    print(snapshots.shape)

    train_set = snapshots[20:101:10]
    train_times = times[20:101:10]

    print(train_set.shape)

    print(train_set.max())
    print(train_set.min())

    if black_box:
        model = Time2Image([1, 64, 256, 512, 4608], [72, 64], cnn_layers=[1, 64, 64], activations=nn.ReLU)
    else:
        model = Time2Image([1, 64, 256, 512, 4608], [72, 64], cnn_layers=[1, 64, 64], activations=nn.GELU)

    model = model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    f = get_f(model, input_scale=1e8, output_scale=500.)

    t = train_times.unsqueeze(1)

    # for e in tqdm(range(WARMUP_EPOCHS)):
    #     optimizer.zero_grad()
    #     prediction = f(t)
    #     loss = loss_fn(prediction, train_set)

    #     loss.backward()
    #     optimizer.step()

    #     if (e+1) % 100 == 0:
    #         print(loss.item())
    
    if black_box:
        torch.save(model.state_dict(), Path("checkpoints") / "time2image_relu_warmup_1ns_20_100.ckp")

    else:
        #torch.save(model.state_dict(), Path("checkpoints") / "time2image_gelu_warmpu_1ns_20_100.ckp")
        model.load_state_dict(torch.load(Path("checkpoints") / "time2image_gelu_warmpu_1ns_20_100.ckp"))
        train_losses = []
        physics_losses = []
        for e in tqdm(range(EPOCHS)):
            optimizer.zero_grad()

            if PINN:
                train_loss, physics_loss = PINN_loss_fn(f, t, train_set, loss_fn, 10, (t[0], t[-1]), geometry)
                loss = train_loss + physics_loss
                train_losses.append(float(train_loss))
                physics_losses.append(float(physics_loss))
            else:
                prediction = f(t)
                loss = loss_fn(prediction, train_set)
                train_losses.append(float(loss))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (e+1) % 100 == 0:
                print(loss.item())

        torch.save(model.state_dict(), Path("checkpoints") / "time2image_gelu_1ns_20_100_1e18.ckp")
        
    # with torch.no_grad():
    #     prediction = f(t)
    
    # plt.semilogy(train_losses, label="train loss")
    # plt.semilogy(physics_losses, label="physics loss")
    # plt.legend()
    # plt.show()

    # save_field_animation(prediction.cpu().squeeze(), None)
    # save_field_animation(mlp_features.cpu().squeeze(), "figures/time2image_gelu_mlp_hidden_state_1ns.gif")

    #preds = f(t)
    #save_field_animation(preds.cpu().detach().numpy(), "figures/time2image_rail_PINN.gif")


if __name__ == "__main__":
    time2Image(True)
    time2Image(False)