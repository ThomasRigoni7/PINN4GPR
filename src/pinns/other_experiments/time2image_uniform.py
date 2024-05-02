import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from kornia.filters import spatial_gradient, gaussian_blur2d
from skimage.measure import block_reduce

from torch.profiler import profile, ProfilerActivity

from src.visualization.field import save_field_animation
from src.pinns.models import Time2Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

PINN = True
DEVICE = "cuda:2"
WARMUP_EPOCHS = 0
EPOCHS = 0

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

    fix, axs = plt.subplots(ncols=4)
    axs[0].imshow(t1.cpu().detach())
    imgs = axs[0].get_images()
    vmin0, vmax0 = imgs[0].get_clim()
    axs[1].imshow(t2.cpu().detach())
    imgs = axs[1].get_images()
    vmin1, vmax1 = imgs[0].get_clim()
    vmin = min(vmin0, vmin1)
    vmax = max(vmax0, vmax1)
    axs[2].imshow((t2 - t1).cpu().detach(), vmin=vmin, vmax=vmax)
    axs[3].imshow(diff.cpu().detach())

    axs[0].set_title("t1")
    axs[1].set_title("t2")
    axs[2].set_title("t2 - t1")
    axs[3].set_title("percentage diff")

    plt.show()
    print(diff.abs().max())
    return torch.all(diff.abs() > tolerance).item()

def calculate_time_derivative_numerically(f, t: torch.Tensor, epsilon: float = 1e-12):

    t_before = t - epsilon
    t_after = t + epsilon

    pred_before = f(t_before)
    pred_after = f(t_after)

    return (pred_after - pred_before) / (2*epsilon)
    
def calculate_2nd_time_derivative_numerically(f, t: torch.Tensor, epsilon: float = 1e-12):

    t_before = t - epsilon
    t_after = t + epsilon

    pred_before = f(t_before)
    pred = f(t)
    pred_after = f(t_after)

    dt0 = (pred - pred_before) / (epsilon)
    dt1 = (pred_after - pred) / (epsilon)

    return (dt1 - dt0) / epsilon

def check_autograd_forward(f, t: torch.Tensor):
    def f_grad_aux(t):
            """
            Auxilliary function to compute first and second order gradients in 1 go
            """
            preds, dft = torch.func.jvp(f, (t,), (torch.ones_like(t),))
            return dft, preds

    dft, dftt, preds = torch.func.jvp(f_grad_aux, (t,), (torch.ones_like(t),), has_aux=True)

    dft_numerical = calculate_time_derivative_numerically(f, t)
    dftt_numerical = calculate_2nd_time_derivative_numerically(f, t)
    for i in range(len(dft)):
        is_close(dft[i], dft_numerical[i])
        is_close(dftt[i], dftt_numerical[i])

    # calculate derivatives in backward fashion to check gradients
    print("calculating backwards derivatives...")
    t.requires_grad_()
    preds2 = f(t)
    print(t.shape)
    print(preds2.shape)
    for y in tqdm(range(preds2.shape[1])):
        for x in tqdm(range(preds2.shape[2]), position=1):
            back_dft = torch.autograd.grad(preds2[:, y, x].unsqueeze(1), t, torch.ones_like(t), create_graph=True)[0]
            print("1st derivative forward:")
            print(dft[:, y, x])
            print("1st derivative backward:")
            print(back_dft.squeeze())
            print("1st derivative numerical:")
            print(dft_numerical[:, y, x])
            # if not torch.allclose(back_dft.squeeze(), dft[:, y, x]):
            #     print()
            #     print("1st derivative WRONG!")
            # back_dftt = torch.autograd.grad(back_dft, t, torch.ones_like(t), create_graph=True)[0]
            # if not torch.allclose(back_dftt.squeeze(), dftt[:, y, x]):
            #     print()
            #     print("2nd derivative WRONG!")

def visualize_pred_and_PDE(f, t: torch.Tensor, geometry:torch.Tensor):
    def f_grad_aux(t):
            """
            Auxilliary function to compute first and second order gradients in 1 go
            """
            preds, dft = torch.func.jvp(f, (t,), (torch.ones_like(t),))
            return dft, preds

    dft, dftt, preds = torch.func.jvp(f_grad_aux, (t,), (torch.ones_like(t),), has_aux=True)

    spatial_gradients = spatial_gradient(preds.unsqueeze(1), mode="sobel", order=2, normalized=True)
    # scale the spatial gradients so that they express the real measurements in meters.
    # 1 cell is 6mm -> 2nd order gradients need to be multiplied by 1/(0.006)**2
    spatial_gradients = spatial_gradients * (1/(0.006)**2)
    print(spatial_gradients.shape)

    spatial_gradients1 = spatial_gradient(preds.unsqueeze(1), mode="sobel", order=1, normalized=True)
    dfx = spatial_gradients1[:, :, 0, :, :]
    dfy = spatial_gradients1[:, :, 1, :, :]
    dfxx = spatial_gradient(dfx, mode="sobel", order=1, normalized=True)[:, :, 0, :, :].squeeze() * (1/(0.006)**2)
    dfyy = spatial_gradient(dfy, mode="sobel", order=1, normalized=True)[:, :, 1, :, :].squeeze() * (1/(0.006)**2)

    # cut the images
    dft = dft[:, 2:-2, 2:-2]
    dftt = dftt[:, 2:-2, 2:-2]
    dfxx = dfxx[:, 2:-2, 2:-2]
    dfyy = dfyy[:, 2:-2, 2:-2]
    geometry = geometry[:, 2:-2, 2:-2]

    epsilon, sigma, mu, _ = geometry

    epsilon = epsilon * EPSILON_0
    mu = mu * MU_0
    sigma = 0.01

    term1 = dftt
    term2 = -(1/(epsilon*mu)) * (dfxx + dfyy)
    term3 = dft * sigma / epsilon

    collocation_loss = term1 + term2 + term3

    vmin = -5e22
    vmax = 5e22

    for i in range(len(term1)):
        fig, axs = plt.subplots(ncols=5)
        axs[0].imshow(preds[i].cpu().detach())
        axs[1].imshow(term1[i].cpu().detach(), vmin=vmin, vmax=vmax)
        axs[2].imshow(term2[i].cpu().detach(), vmin=vmin, vmax=vmax)
        axs[3].imshow((dft/epsilon)[i].cpu().detach(), vmin=vmin, vmax=vmax)
        mappable = axs[4].imshow(collocation_loss[i].cpu().detach(), vmin=vmin, vmax=vmax)

        axs[0].set_title("Prediction")
        axs[1].set_title("dftt term")
        axs[2].set_title("dfxx/dfyy term")
        axs[3].set_title("dft term")
        axs[4].set_title("total")

        divider = make_axes_locatable(axs[4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(mappable, cax=cax)
        plt.show()



def PINN_loss_fn(f: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor, labels: torch.Tensor, loss_fn: Callable, n_collocations: int, coll_domain: tuple[float, float], geometry: torch.Tensor):

    def f_grad_aux(t):
            """
            Auxilliary function to compute first and second order gradients in 1 go
            """
            preds, dft = torch.func.jvp(f, (t,), (torch.ones_like(t),))
            return dft, preds

    # Observation loss:
    preds = f(t)
    observation_loss = loss_fn(preds, labels)

    ###
    collocation_points = torch.rand((20, 1), device=DEVICE)
    collocation_points = collocation_points * (4e-9) + 2e-9
    collocation_points.requires_grad_()
    collocation_points = collocation_points.sort(dim=0).values

    # print("Collocation points:", collocation_points.squeeze())
    # input()

    # how to compute gradient w.r.t output image?
    # use torch.func.jvp: it is forward automatic differentiation. 1 input -> multiple outputs
    dft, dftt, coll_preds = torch.func.jvp(f_grad_aux, (collocation_points,), (torch.ones_like(collocation_points),), has_aux=True)

    # scale the spatial gradients so that they
    # express the real measurements in meters.
    # 1 cell is 6mm -> 2nd order gradients need to be multiplied by 1/(0.006)**2
    spatial_gradients1 = spatial_gradient(coll_preds.unsqueeze(1), mode="sobel", order=1, normalized=True)
    dfx = spatial_gradients1[:, :, 0, :, :]
    dfy = spatial_gradients1[:, :, 1, :, :]
    dfxx = spatial_gradient(dfx, mode="sobel", order=1, normalized=True)[:, :, 0, :, :].squeeze() * (1/(0.006)**2)
    dfyy = spatial_gradient(dfy, mode="sobel", order=1, normalized=True)[:, :, 1, :, :].squeeze() * (1/(0.006)**2)

    # cut the images
    dft = dft[:, 2:-2, 2:-2]
    dftt = dftt[:, 2:-2, 2:-2]
    dfxx = dfxx[:, 2:-2, 2:-2]
    dfyy = dfyy[:, 2:-2, 2:-2]
    geometry = geometry[:, 2:-2, 2:-2]

    epsilon, sigma, mu, _ = geometry
    epsilon = epsilon * EPSILON_0
    mu = mu * MU_0

    term1 = dftt
    term2 = -(1/(epsilon*mu)) * (dfxx + dfyy)
    term3 = dft * sigma / epsilon
    # term3 = 0

    collocation_loss = term1 + term2 + term3
    collocation_loss = loss_fn(1e-22 * collocation_loss, torch.zeros_like(collocation_loss))
    physics_loss = collocation_loss

    return observation_loss, physics_loss


def get_f(model: Time2Image, input_scale: float, output_scale: float):
    input_scale = torch.tensor(input_scale)
    output_scale = torch.tensor(output_scale)
    def f(t):
        t = t * input_scale
        out = model(t)
        out = out.squeeze() * output_scale
        out = out[:, 2:-2, 3:-3]
        return out
    return f

def time2Image():

    # geometry = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00000/scan_00000_geometry.npy")
    # snapshots = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00000/snapshots.npz")["00000_E"]

    geometry = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00001/scan_00001_geometry.npy")
    snapshots = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00001/snapshots.npz")["00000_E"]

    geometry = block_reduce(geometry, block_size=(1, 3, 3), func=np.mean)
    geometry = torch.from_numpy(geometry).to(DEVICE)

    # snapshots = np.load("paper_data/2layer_wavefield.npz")["00000_E"]
    #snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]

    snapshots = torch.from_numpy(snapshots).to(DEVICE, dtype=torch.float32)
    #times = np.load("paper_data/uniform_wavefield.npz")["0000_times"]
    times = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00001/snapshots.npz")["00000_times"]
    times = torch.from_numpy(times).to(DEVICE, dtype=torch.float32)

    train_set = snapshots[20:40:5]
    save_field_animation(train_set.cpu().numpy(), "figures/time2image_rail_label.gif", interval=100)
    train_times = times[20:40:5]
    test_times = times[20:60:2].unsqueeze(1)
    print("train times:", train_times)
    print("test times:", test_times.squeeze())

    print(train_set.max())
    print(train_set.min())

    model = Time2Image([1, 64, 256, 512, 4608], [72, 64], cnn_layers=[1, 64, 64], activations=nn.GELU)

    model = model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    f = get_f(model, input_scale=1e8, output_scale=500.)

    t = train_times.unsqueeze(1)

    # model.load_state_dict(torch.load(Path("checkpoints") / "time2image_gelu_warmup_05ns_uniform.ckp"))
    # check_autograd_forward(f, t)
    

    for e in tqdm(range(WARMUP_EPOCHS)):
        optimizer.zero_grad()
        prediction = f(t)
        loss = loss_fn(prediction, train_set)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if (e+1) % 100 == 0:
            print(loss.item())
    
    # torch.save(model.state_dict(), Path("checkpoints") / "time2image_gelu_warmup_02ns.ckp")
    # torch.save(model.state_dict(), Path("checkpoints") / "time2image_gelu_warmup_20_40_5_uniform.ckp")
    # model.load_state_dict(torch.load(Path("checkpoints") / "time2image_gelu_warmup_20_40_5_uniform_.ckp"))
    # visualize_pred_and_PDE(f, test_times, geometry)

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
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if (e+1) % 100 == 0:
            print(loss.item())
        
    model.load_state_dict(torch.load(Path("checkpoints") / "time2image_gelu_finetune_20_40_5_uniform_finetune.ckp"))
    with torch.no_grad():
        prediction = f(test_times)
    
    plt.semilogy(train_losses, label="train loss")
    plt.semilogy(physics_losses, label="physics loss")
    plt.legend()
    plt.show()
    
    # torch.save(model.state_dict(), Path("checkpoints") / "time2image_gelu_finetune_20_40_5_uniform_finetune.ckp")

    save_field_animation(prediction.cpu().squeeze(), "figures/time2image_gelu_predictions_PINN.gif")
    # save_field_animation(prediction.cpu().squeeze(), "figures/time2image_gelu_predictions_1ns.gif")
    # save_field_animation(mlp_features.cpu().squeeze(), "figures/time2image_gelu_mlp_hidden_state_1ns.gif")

    #preds = f(t)
    #save_field_animation(preds.cpu().detach().numpy(), "figures/time2image_rail_PINN.gif")
    visualize_pred_and_PDE(f, test_times, geometry)


if __name__ == "__main__":
    time2Image()