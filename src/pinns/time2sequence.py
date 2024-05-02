"""
This module contains the implementation of the 1D wave experiment:

Three models (MLP, CNN and discrete-output MLP) are trained on the same data composed of 
a 1D wave propagating through time from left to right. Both black box and physics-informed
models are trained on each architecture.

The MLP shows the ability to perform sparse reconstruction and domain extension, while both
the CNN and discrete MLP models fail at it.
"""

import numpy as np
import torch
import torch.nn as nn
from src.pinns.models.time2sequence import MLP, Time2Sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
from kornia.filters import spatial_gradient
import matplotlib.animation as animation
from pathlib import Path

DEVICE = "cuda:0"
OBS_WAVE_SIZE = 512
WAVE_INTERPOLATION_SIZE = 200
WAVE_SPEED = 12

class DivisibleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_mlp = MLP(1, [256]*5, 512, nn.SiLU)
        self.t_mlp = MLP(1, [256]*5, 512, nn.SiLU)

    def forward(self, x, t):
        v_x = self.x_mlp(x)
        v_t = self.t_mlp(t)
        out = torch.bmm(v_x.unsqueeze(-2), v_t.unsqueeze(-1))
        return out

def show_wave_evolution(test_pred, output_path=None):

    fig, ax = plt.subplots()
    line, = ax.plot(test_pred[0].cpu())
    # plt.ylim(-1, 1)

    def animate(i):
        index = i % len(test_pred)
        line.set_ydata(test_pred[index].cpu())  # update the data.
        return line,


    ani = animation.FuncAnimation(
        fig, animate, interval=100, blit=True, frames=len(test_pred))
    
    if output_path:
        ani.save(output_path)
    else:
        plt.show()

def build_obs_wave():
    snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]
    labels = np.zeros((5, OBS_WAVE_SIZE), dtype=np.float32)
    wave = snapshots[30][100:, 100]
    wave /= -wave.min()
    wave = wave.squeeze()

    wave = np.interp(np.arange(WAVE_INTERPOLATION_SIZE), np.arange(100) * (WAVE_INTERPOLATION_SIZE / len(wave)), wave)

    labels[0][:WAVE_INTERPOLATION_SIZE] = wave
    labels[1][WAVE_SPEED:WAVE_INTERPOLATION_SIZE + WAVE_SPEED] = wave
    labels[2][2*WAVE_SPEED:WAVE_INTERPOLATION_SIZE + 2*WAVE_SPEED] = wave
    labels[3][3*WAVE_SPEED:WAVE_INTERPOLATION_SIZE + 3*WAVE_SPEED] = wave
    labels[4][4*WAVE_SPEED:WAVE_INTERPOLATION_SIZE + 4*WAVE_SPEED] = wave

    from scipy.ndimage import gaussian_filter1d
    labels[0] = gaussian_filter1d(labels[0], 2)
    labels[1] = gaussian_filter1d(labels[1], 2)
    labels[2] = gaussian_filter1d(labels[2], 2)
    labels[3] = gaussian_filter1d(labels[3], 2)
    labels[4] = gaussian_filter1d(labels[4], 2)

    plt.plot(labels[0])
    plt.plot(labels[1])
    plt.plot(labels[2])
    plt.plot(labels[3])
    plt.plot(labels[4])
    plt.show()

    return torch.from_numpy(labels).to(DEVICE)

def build_mlp_dataset(obs_waves: torch.Tensor, times: torch.Tensor):
    assert len(obs_waves) == len(times)
    inputs = []
    labels = []
    for wave, time in zip(obs_waves, times):
        for x, val in enumerate(wave):
            inputs.append((float(x), float(time)))
            labels.append(float(val))
        # inputs.append(torch.stack([torch.arange(0.0, len(wave), 1.0), time.broadcast_to(len(wave))]).T)
        # labels.append(wave)
    
    inputs = torch.asarray(inputs)
    labels = torch.asarray(labels)

    inputs = inputs.to(DEVICE).requires_grad_()
    labels = labels.to(DEVICE)

    return inputs, labels


def train_mlp_pinn(black_box: bool):
    model_str = "NN" if black_box else "PINN"
    results_folder = f"results/time2sequence/new/mlp/{model_str}_{OBS_WAVE_SIZE}_{WAVE_INTERPOLATION_SIZE}_{WAVE_SPEED}/"
    results_folder = Path(results_folder)
    results_folder.mkdir(exist_ok=True, parents=True)

    model = MLP(2, [256]*5, 1, nn.SiLU)

    model = model.to(DEVICE)

    times = torch.arange(0.0, 4.1, 1.0)
    waves = build_obs_wave()
    x_obs, obs_label = build_mlp_dataset(waves, times)

    show_wave_evolution(waves, results_folder / "training_set.gif")

    def f(x:torch.Tensor, t:torch.Tensor):
        x = x / 500
        t = t / 10

        input = torch.stack([x, t], dim=-1)
        out = model(input)

        # x, t = x.unsqueeze(-1), t.unsqueeze(-1)
        # out = model(x, t)
        
        return out.squeeze()
    
    def loss_PINN(f, x_obs):

        obs_preds = f(x_obs[:, 0], x_obs[:, 1])
        obs_loss = regular_loss(obs_preds.squeeze(), obs_label.squeeze())

        physics_loss = 0

        xc = torch.rand((2000, 1), device=DEVICE) * OBS_WAVE_SIZE
        tc = torch.rand((2000, 1), device=DEVICE) * 40.
        xc.requires_grad_()
        tc.requires_grad_()
        
        uc = f(xc, tc)

        dfx = torch.autograd.grad(uc, xc, torch.ones_like(uc), create_graph=True)[0]
        dft = torch.autograd.grad(uc, tc, torch.ones_like(uc), create_graph=True)[0]
        dftt = torch.autograd.grad(dft, tc, torch.ones_like(dft), create_graph=True)[0]
        dfxx = torch.autograd.grad(dfx, xc, torch.ones_like(dfx), create_graph=True)[0]

        physics_remainder = (dftt - (WAVE_SPEED**2)*(dfxx))
        physics_loss = regular_loss(physics_remainder, torch.zeros_like(physics_remainder))

        return obs_loss, physics_loss


    regular_loss = nn.MSELoss()
    from src.pinns.paper.model import L4loss
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    train_loss_step = []
    physics_loss_step = []
    EPOCHS = 5000
    for e in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        preds = f(x_obs[:, 0], x_obs[:, 1])
        l = L4loss(preds.squeeze(), obs_label.squeeze())
        l.backward()
        optimizer.step()
        train_loss_step.append(float(l))
        physics_loss_step.append(0.)

    optimizer = torch.optim.Adam(model.parameters(), 0.0001)

    for e in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        if black_box:
            preds = f(x_obs[:, 0], x_obs[:, 1])
            l = regular_loss(preds, obs_label)
            tl = l
            pl = 0
        else:
            tl, pl = loss_PINN(f, x_obs)
            l = tl + pl
        l.backward()
        optimizer.step()
        train_loss_step.append(float(tl))
        physics_loss_step.append(float(pl))

    plt.close("all")
    plt.semilogy(train_loss_step, label="train loss")
    plt.semilogy(physics_loss_step, label="physics loss")
    plt.legend()
    plt.savefig(results_folder / "train_loss.png")
    plt.close()

    torch.save(model.state_dict(), results_folder / "model.ckp")
    model.load_state_dict(torch.load(results_folder / "model.ckp"))

    x_test = torch.arange(0., OBS_WAVE_SIZE, 1.0)
    t_test = torch.arange(0., 40., 0.2)
    t_test, x_test = torch.meshgrid([t_test, x_test], indexing="ij")

    x_test = x_test.to(DEVICE)
    t_test = t_test.to(DEVICE)

    print("Min train loss:", min(train_loss_step))
    print("Max train loss:", max(train_loss_step))
    print("Min physics loss:", min(physics_loss_step))
    print("Max physics loss:", max(physics_loss_step))

    test_preds = f(x_test, t_test)


    show_wave_evolution(test_preds.cpu().detach(), results_folder / "predictions.gif")



def train_time2seq(use_mlp: bool, black_box: bool):
    model_name = "expanding_mlp" if use_mlp else "cnn"
    model_str = "NN" if black_box else "PINN"
    results_folder = f"results/time2sequence/new/{model_name}/{model_str}_{OBS_WAVE_SIZE}_{WAVE_INTERPOLATION_SIZE}_{WAVE_SPEED}/"
    results_folder = Path(results_folder)
    results_folder.mkdir(exist_ok=True, parents=True)

    if use_mlp:
        model = MLP(1, [64, 256], OBS_WAVE_SIZE, nn.SiLU)
    else:
        model = Time2Sequence(cnn_layers=[1, 64], activations=nn.SiLU)
        model.gaussian_kernel = model.gaussian_kernel.to(DEVICE)

    model = model.to(DEVICE)

    def f(t: torch.Tensor):
        t = t / 10
        out = model(t)
        return out.squeeze()

    t_train = torch.arange(0.0, 4.1, 1.0, requires_grad=True, device=DEVICE).unsqueeze(-1)
    obs_label = build_obs_wave()

    t_test = torch.arange(0., 15., 0.05, device=DEVICE, requires_grad=True).unsqueeze(1)


    def loss_PINN(f, t_obs):

        obs_preds = f(t_obs)
        obs_loss = regular_loss(obs_preds, obs_label.squeeze())

        def f_grad_aux(t: torch.Tensor):
            y, grad = torch.func.jvp(f, (t,), (tangent,))
            return grad, y
        
        t_c = torch.rand((1000, 1), device=DEVICE) * 40.
        t_c = t_c.sort(dim=0).values

        
        tangent = torch.ones_like(t_c, device=DEVICE, requires_grad=True)
        dft, dftt, coll_preds = torch.func.jvp(f_grad_aux, (t_c,), (torch.ones_like(t_c),), has_aux=True)



        coll_preds = coll_preds.unsqueeze(1)

        spatial_gradients1 = spatial_gradient(coll_preds.unsqueeze(1), mode="sobel", order=1, normalized=True)
        dfx = spatial_gradients1[:, :, 0, :, :]
        dfxx = spatial_gradient(dfx, mode="sobel", order=1, normalized=True)[:, :, 0, :, :].squeeze()

        physics_remainder2 = (1/WAVE_SPEED)*(dftt - (WAVE_SPEED**2)*(dfxx))
        physics_loss = regular_loss(physics_remainder2, torch.zeros_like(physics_remainder2))

        return obs_loss, physics_loss

    regular_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    EPOCHS = 10000

    train_loss_step = []
    physics_loss_step = []
    for e in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        pred = f(t_train)
        l = regular_loss(pred, obs_label.squeeze())
        l.backward()
        optimizer.step()
        train_loss_step.append(float(l))
        physics_loss_step.append(float(0))

    for e in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        if black_box:
            pred = f(t_train)
            l = regular_loss(pred, obs_label.squeeze())
            tl = l
            pl = 0
        else:
            tl, pl = loss_PINN(f, t_train)
            l = tl + pl
        l.backward()
        optimizer.step()
        train_loss_step.append(float(tl))
        physics_loss_step.append(float(pl))

    plt.close("all")
    plt.semilogy(train_loss_step, label="train loss")
    plt.semilogy(physics_loss_step, label="physics loss")
    plt.legend()
    plt.savefig(results_folder / "train_loss.png")
    plt.close()
    
    torch.save(model.state_dict(), results_folder / "model.ckp")
    model.load_state_dict(torch.load(results_folder / "model.ckp"))


    with torch.no_grad():
        test_pred = f(t_test)
    

    show_wave_evolution(t_train.detach().cpu(), results_folder / "train_set.gif")
    show_wave_evolution(test_pred, results_folder / "predictions.gif")

    spatial_gradients1 = spatial_gradient(test_pred.unsqueeze(1).unsqueeze(1), mode="sobel", order=1, normalized=True)
    dfx = spatial_gradients1[:, :, 0, :, :]
    dfxx = spatial_gradient(dfx, mode="sobel", order=1, normalized=True)[:, :, 0, :, :].squeeze()

    show_wave_evolution(dfxx, results_folder / "dfxx.gif")


def check_numerical_derivative():
    wave = build_obs_wave()
    print(wave.shape)

    spatial_gradients1 = spatial_gradient(wave.unsqueeze(1).unsqueeze(1), mode="sobel", order=1, normalized=True)
    dfx = spatial_gradients1[:, :, 0, :, :]
    dfxx = spatial_gradient(dfx, mode="sobel", order=1, normalized=True)[:, :, 0, :, :].squeeze()

    show_wave_evolution(dfxx, None)

if __name__ == "__main__":
    
    train_mlp_pinn(True)
    train_mlp_pinn(False)
    train_time2seq(False, True)
    train_time2seq(True, True)
    train_time2seq(False, False)
    train_time2seq(True, False)
