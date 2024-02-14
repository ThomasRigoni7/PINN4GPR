from typing import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from src.pinns.paper.dataset import MyNormalizer

IMG_SIZE = (200, 200)
SPATIAL_DOMAIN_SIZE = ((0, 20), (0, 20))
EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6

class MLP(nn.Module):
    def __init__(self, num_inputs: int, hidden_layer_sizes: list[int], num_outputs: int, activation: nn.Module=nn.SiLU):
        super().__init__()
        if len(hidden_layer_sizes) < 1:
            raise Exception("MLP needs to have at least 1 hidden layer!")
        
        self.activation = activation()
        self.input_layer = nn.Linear(num_inputs, hidden_layer_sizes[0])
        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], num_outputs)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, y, t):
        x = torch.stack([x, y, t])
        if x.ndim == 1:
            x = x[:, None]
        x = x.transpose(1, 0)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x

def get_f(model: MLP, scaler:MyNormalizer) -> Callable:
    def f(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x, y, t, _ = scaler.transform_(x, y, t, None)

        u : torch.Tensor = model(x, y, t)
        u.squeeze()

        _, u = scaler.inverse_transform(None, u)
        return u.squeeze()

    return f

def get_EM_values(x: torch.Tensor, y: torch.Tensor, geometry: torch.Tensor):
    """
    Returns the EM values of the point, given its coordinates and the geometry
    """
    percent_x = (x - SPATIAL_DOMAIN_SIZE[0][0]) / (SPATIAL_DOMAIN_SIZE[0][1] - SPATIAL_DOMAIN_SIZE[0][0])
    index_x = percent_x * (IMG_SIZE[0])
    percent_y = (y - SPATIAL_DOMAIN_SIZE[1][0]) / (SPATIAL_DOMAIN_SIZE[1][1] - SPATIAL_DOMAIN_SIZE[1][0])
    index_y = percent_y * (IMG_SIZE[1])

    index_x = torch.clamp(index_x, 0, IMG_SIZE[0] - 1)
    index_y = torch.clamp(index_y, 0, IMG_SIZE[1] - 1)

    return geometry[:, index_y.int(), index_x.int()]

def L4loss(pred: torch.Tensor, target: torch.Tensor):
    assert pred.shape == target.shape
    return ((pred - target)**4).mean()

def L6loss(pred: torch.Tensor, target: torch.Tensor):
    assert pred.shape == target.shape
    return ((pred - target)**6).mean()
    
def get_PINN_warmup_loss_fn(training_points_loss_fn):
    def loss_fn(f, x, y, t, u,  
                boundary_points: torch.Tensor,
                collocation_points_xyt: torch.Tensor, 
                geometry: np.ndarray):
        """
        Loss function for the network:
        
        Parameters
        ----------
        `x`, `y`, and `t` are the inputs to the network, `u` is the output electric field.
        
        `domain_size` is the time and spatial size of the domain in shape [t, y, x], in
        where to compute the physics (collocation) loss.
        """

        # training points:
        train_preds = f(x, y, t)
        train_loss = training_points_loss_fn(train_preds, u)
        return train_loss, 0

    return loss_fn

def get_PINN_uniform_loss_fn(training_points_loss_fn):
    def loss_fn(f, x, y, t, u,  
                boundary_points: torch.Tensor,
                collocation_points_xyt: torch.Tensor, 
                geometry: np.ndarray):
        """
        Loss function for the network:
        
        Parameters
        ----------
        `x`, `y`, and `t` are the inputs to the network, `u` is the output electric field.
        
        `domain_size` is the time and spatial size of the domain in shape [t, y, x], in
        where to compute the physics (collocation) loss.
        """

        # training points:
        train_preds = f(x, y, t)
        train_loss = training_points_loss_fn(train_preds, u)

        # collocation points
        l = nn.MSELoss()
        xc, yc, tc = collocation_points_xyt
        EM_values = get_EM_values(xc, yc, geometry)

        epsilon, sigma, mu, _ = EM_values
        epsilon *= EPSILON_0
        mu *= MU_0

        xc.requires_grad_()
        tc.requires_grad_()
        yc.requires_grad_()
        uc = f(xc, yc, tc)

        # Calculate first and second derivatives:
        # The derivatives need to require gradient, so we need to set create_graph.
        # For some reason, 'retain_graph' is not ok for the second derivatives and makes the network diverge
        dfx = torch.autograd.grad(uc, xc, torch.ones_like(uc), create_graph=True)[0]
        dfy = torch.autograd.grad(uc, yc, torch.ones_like(uc), create_graph=True)[0]
        dft = torch.autograd.grad(uc, tc, torch.ones_like(uc), create_graph=True)[0]
        dftt = torch.autograd.grad(dft, tc, torch.ones_like(dft), create_graph=True)[0]
        dfxx = torch.autograd.grad(dfx, xc, torch.ones_like(dfx), create_graph=True)[0]
        dfyy = torch.autograd.grad(dfy, yc, torch.ones_like(dfy), create_graph=True)[0]

        term1 = dftt
        term2 = (1/(epsilon*mu)) * (dfxx + dfyy)
        term3 = dft * sigma / epsilon

        # collocation_loss = dftt - (1/(epsilon*mu)) * (dfxx + dfyy) + dft * sigma / epsilon
        collocation_loss = term1 - term2 + term3
        collocation_loss = l(2e-17 * collocation_loss, torch.zeros_like(collocation_loss))

        physics_loss = collocation_loss

        return train_loss, physics_loss

    return loss_fn

def get_PINN_loss_fn(training_points_loss_fn):
    def loss_fn(f, x, y, t, u,  
                boundary_points: torch.Tensor,
                collocation_points_xyt: torch.Tensor, 
                geometry: np.ndarray):
        """
        Loss function for the network:
        
        Parameters
        ----------
        `x`, `y`, and `t` are the inputs to the network, `u` is the output electric field.
        
        `domain_size` is the time and spatial size of the domain in shape [t, y, x], in
        where to compute the physics (collocation) loss.
        """

        # training points:
        train_preds = f(x, y, t)
        train_loss = training_points_loss_fn(train_preds, u)

        l = nn.MSELoss()
        # # boundary conditions:
        # boundary_points_xmin = boundary_points[0]
        # boundary_points_ymin = boundary_points[1]
        # boundary_points_xmax = boundary_points[2]
        # boundary_points_ymax = boundary_points[3]
        

        # free-surface: favours a reflection at the position it is calculated
        # lapl(f(x, y=0, t)) = 0
        # xb, yb, tb = boundary_points

        # # split each point into 2, one in each medium
        # yb0 = yb - 0.001
        # yb1 = yb + 0.001

        # epsilon_r0, _, mu_r0, _ = get_EM_values(xb, yb0, geometry)
        # epsilon_r1, _, mu_r1, _ = get_EM_values(xb, yb1, geometry)

        # # calculate the speed of the EM waves in the two mediums
        # v0 = torch.sqrt(1 / (epsilon_r0 * EPSILON_0 * mu_r0* MU_0))
        # v1 = torch.sqrt(1 / (epsilon_r1 * EPSILON_0 * mu_r1* MU_0))

        # speed_factor = v0 / v1


        # xb.requires_grad_()
        # yb0.requires_grad_()
        # yb1.requires_grad_()
        # tb.requires_grad_()
        # ub0 = f(xb, yb0, tb)
        # ub1 = f(xb, yb1, tb)
        # # dft0 = torch.autograd.grad(ub0, tb, torch.ones_like(ub0), create_graph=True)[0]
        # # dft1 = torch.autograd.grad(ub1, tb, torch.ones_like(ub1), create_graph=True)[0]
        # boundary_loss_field = ub0 - ub1
        # # boundary_loss_derivative = 1e-10 * (dft0 - dft1)
        # # boundary_loss = l(boundary_loss_field, torch.zeros_like(boundary_loss_field)) + \
        # #     l(boundary_loss_derivative, torch.zeros_like(boundary_loss_derivative))
        # boundary_loss = l(boundary_loss_field, torch.zeros_like(boundary_loss_field))
        boundary_loss = 0

        # collocation points:
        # d2f_dt2 - 1/(mu*eps) * (d2f_dx2 + d2f_dy2) + (sigma/epsilon)*df_dt = 0
        xc, yc, tc = collocation_points_xyt
        EM_values = get_EM_values(xc, yc, geometry)
        epsilon, sigma, mu, _ = EM_values

        epsilon *= EPSILON_0
        mu *= MU_0


        xc.requires_grad_()
        tc.requires_grad_()
        yc.requires_grad_()
        uc = f(xc, yc, tc)

        # Calculate first and second derivatives:
        # The derivatives need to require gradient, so we need to set create_graph.
        # For some reason, 'retain_graph' is not ok for the second derivatives and makes the network diverge
        dfx = torch.autograd.grad(uc, xc, torch.ones_like(uc), create_graph=True)[0]
        dfy = torch.autograd.grad(uc, yc, torch.ones_like(uc), create_graph=True)[0]
        dft = torch.autograd.grad(uc, tc, torch.ones_like(uc), create_graph=True)[0]
        dftt = torch.autograd.grad(dft, tc, torch.ones_like(dft), create_graph=True)[0]
        dfxx = torch.autograd.grad(dfx, xc, torch.ones_like(dfx), create_graph=True)[0]
        dfyy = torch.autograd.grad(dfy, yc, torch.ones_like(dfy), create_graph=True)[0]

        term1 = dftt
        term2 = -(1/(epsilon*mu)) * (dfxx + dfyy)
        term3 = dft * sigma / epsilon

        # print("term1:", term1)
        # print("term2:", term2)
        # print("term3:", term3)


        # collocation_loss = dftt - (1/(epsilon*mu)) * (dfxx + dfyy) + dft * sigma / epsilon
        collocation_loss = term1 + term2 + term3
        collocation_loss = l(2e-18 * collocation_loss, torch.zeros_like(collocation_loss))


        # physics_loss = l(physics_loss, torch.zeros_like(physics_loss)) + l(boundary_loss, torch.zeros_like(boundary_loss))
        # print("boundary loss:", boundary_loss)
        # print("collocation loss:", collocation_loss)
        physics_loss = collocation_loss + boundary_loss

        return train_loss, physics_loss

    return loss_fn

def predict(model: MLP, samples: torch.Tensor):
    model.eval()
    x, y, t, u = samples
    predictions = model(x, y, t).cpu().detach().squeeze()
    
    return predictions

def predict_functional(f, samples: torch.Tensor):
    x, y, t, u = samples
    with torch.no_grad():
        predictions = f(x, y, t).cpu().detach().squeeze()
    
    return predictions

def evaluate_functional(f: Callable, samples: torch.Tensor, loss_fn):
    x, y, t, u = samples
    with torch.no_grad():
        loss = loss_fn(f(x, y, t), u).cpu().detach()
    
    return float(loss.mean())

def evaluate(model: MLP, samples: torch.Tensor, regular_loss_fn):
    model.eval()
    x, y, t, u = samples
    loss = regular_loss_fn(model(x, y, t), u.view(-1, 1)).cpu().detach()
    
    return float(loss.mean())

def show_field(img: torch.Tensor, ax: Axes = None, vmin=None, vmax=None):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.shape != IMG_SIZE:
        img = img.reshape(IMG_SIZE)
    if ax is not None:
        mappable = ax.imshow(img, vmin=vmin, vmax=vmax)
        imgs = ax.get_images()
        vmin, vmax = imgs[0].get_clim()
        return mappable, vmin, vmax
    else:
        plt.imshow(img)
        plt.show()

def show_predictions(f_PINN: Callable, f_regular: Callable, samples: torch.Tensor, save_path: str | Path = None):
    # show predictions of the field for NN and PINN
    ground_truth = samples[3].reshape(IMG_SIZE).cpu()
    regular_predictions =  predict_functional(f_regular, samples)
    regular_predictions = regular_predictions.reshape(IMG_SIZE)
    PINN_predictions =  predict_functional(f_PINN, samples)
    PINN_predictions = PINN_predictions.reshape(IMG_SIZE)

    fig, axs = plt.subplots(nrows=2, ncols=3)
    mappable, vmin, vmax = show_field(ground_truth, axs[0][1])
    axs[0][1].set_title("ground truth")
    show_field(regular_predictions, axs[0][0], vmin, vmax)
    axs[0][0].set_title("NN predictions")
    show_field(PINN_predictions, axs[0][2], vmin, vmax)
    axs[0][2].set_title("PINN_predictions")
    show_field(regular_predictions - ground_truth, axs[1][0], vmin, vmax)
    axs[1][0].set_title("NN - GT")
    show_field(regular_predictions - PINN_predictions, axs[1][1], vmin, vmax)
    axs[1][1].set_title("NN - PINN")
    show_field(PINN_predictions - ground_truth, axs[1][2], vmin, vmax)
    axs[1][2].set_title("PINN - GT")
    fig.colorbar(mappable, ax=axs, location="right", shrink=0.7)
    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    fig.clear()
    plt.close(fig)