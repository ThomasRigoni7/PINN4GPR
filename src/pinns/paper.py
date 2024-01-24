"""
This module contains the implementation of some of the results shown in the paper.
https://library.seg.org/doi/10.1190/geo2022-0293.1
"""

from collections import OrderedDict
import math
from tqdm import tqdm
import copy
from typing import Callable
import cv2

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchopt

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from src.visualization.misc import save_field_animation
IMG_SIZE = (200, 200)
SPATIAL_DOMAIN_SIZE = ((0, 20), (0, 20))
COLLOCATION_DOMAIN_SIZE = (40e-9, 20, 20)
DEVICE = "cuda:2"
RNG = np.random.default_rng(42)
N_COLLOCATION_POINTS = 40000
N_BOUNDARY_POINTS = 1250
LR = 0.001
EPOCHS_WARMUP = 10000
EPOCHS = 3000
LOGGING_FREQUENCY = 1000

EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6

class MyNormalizer():
    def __init__(self):
        pass

    def fit(self, data: torch.Tensor, labels: torch.Tensor):
        """
        data: 2D tensor of shape [N samples, N features], order is x, y, t
        labels: 1D tensor of labels
        """
        self.data_scale = data.abs().max(dim=0).values
        self.label_scale = labels.abs().max()

        print("Data scale:", self.data_scale)
        print("Label scale:", self.label_scale)
    
    def transform(self, data: torch.Tensor | None, labels: torch.Tensor | None):
        """
        data: 2D tensor of shape [N samples, N features], order is x, y, t
        labels: 1D tensor of labels
        """
        if data is not None:
            data = data / self.data_scale
        if labels is not None: 
            labels = data / self.label_scale
        return data, labels
    
    def transform_(self, x: torch.Tensor | None, y: torch.Tensor | None, t: torch.Tensor | None, labels: torch.Tensor | None):
        """
        all the input tensors are 1D
        """
        x_scale, y_scale, t_scale = self.data_scale
        if x is not None:
            x = x / x_scale
        if y is not None:
            y = y / y_scale
        if t is not None:
            t = t / t_scale
        if labels is not None: 
            labels = labels / self.label_scale
        return x, y, t, labels
    
    def inverse_transform(self, data: torch.Tensor | None, labels: torch.Tensor | None):
        """
        data: 2D tensor of shape [N samples, N features]
        labels: 1D tensor of labels
        """
        if data is not None:
            data = data * self.data_scale
        if labels is not None: 
            labels = labels *self.label_scale
        return data, labels
    
    def inverse_transform_(self, x: torch.Tensor | None, y: torch.Tensor | None, t: torch.Tensor | None, labels: torch.Tensor | None):
        """
        all the input tensors are 1D
        """
        x_scale, y_scale, t_scale = self.data_scale
        if x is not None:
            x = x *x_scale
        if y is not None:
            y = y * y_scale
        if t is not None:
            t = t * t_scale
        if labels is not None: 
            labels = labels * self.label_scale
        return x, y, t, labels

    def _scale_power(self, labels: torch.Tensor, power: float):
        negative = labels < 0
        labels = labels.abs()
        labels = torch.pow(labels, power)
        labels[negative] *= -1
        return labels

class PaperDataset(Dataset):
    def __init__(self, snapshots: np.ndarray, t_offsets: list[float], scaler: MyNormalizer = None):
        # input array has shape [t, y, x] -> gets flattened in x -> y -> t
        self.snapshots_shape = snapshots.shape
        self.snapshots = snapshots.flatten()
        self.t_offsets = t_offsets
        data = []
        labels = []
        for index in range(len(self.snapshots)):
            x = index % self.snapshots_shape[2] / 10
            y = (index // self.snapshots_shape[2]) % self.snapshots_shape[1] / 10
            t = self.t_offsets[index // (self.snapshots_shape[1] * self.snapshots_shape[2])] * 1e-9
            u = self.snapshots[index]
            data.append((x, y, t))
            labels.append(u)
        data = np.array(data)
        labels = np.array(labels)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        # labels = self.scale_power(labels, 1/3)
        if scaler is None:
            self.scaler = MyNormalizer()
            self.scaler.fit(self.data, self.labels)
        else:
            self.scaler = scaler

    

    def __len__(self):
        return math.prod(self.snapshots_shape)
    
    def __getitem__(self, index):
        """
        Iteration is done in x -> y -> t order
        """
        d = self.data[index]
        return d[0], d[1], d[2], self.labels[index]
        #return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(t, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)

    def get_frame(self, index: int):
        """
        Returns all the points associated with a specific frame

        Parameters
        ----------
        index : int
            index of the frame to return
        
        Returns
        -------
        Tensor
            all the points related to the frame, in image shape of [height, width]
        """
        start_index = self.snapshots_shape[-1] * self.snapshots_shape[-2] * index
        points = []
        for i in range(start_index, start_index + self.snapshots_shape[-1] * self.snapshots_shape[-2]):
            p = self[i]
            points.append(p)
        points = np.array(points).transpose(1, 0)
        points = points.reshape(4, self.snapshots_shape[-2], self.snapshots_shape[-1])
        return points
        #     points.append(p[3])
        # return torch.tensor(points).reshape(self.snapshots_shape[-2], self.snapshots_shape[-1])

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

def get_f_and_derivatives(model, scaler:MyNormalizer):

    def f(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, params: dict[str, torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
        
        # the functional optimizer works with parameters represented as a tuple instead
        # of the dictionary form required by the `functional_call` API 
        # here we perform the conversion from tuple to dictionary
        if isinstance(params, tuple):
            params_dict = tuple_to_dict_parameters(model, params)
        else:
            params_dict = params

        x, y, t, _ = scaler.transform_(x, y, t, None)

        u : torch.Tensor = functional_call(model, params_dict, (x, y, t))
        u.squeeze()

        _, u = scaler.inverse_transform(None, u)
        return u.squeeze()
    
    df_dx = grad(f, 0)
    d2f_dx2 = grad(df_dx, 0)

    df_dy = grad(f, 1)
    d2f_dy2 = grad(df_dy, 1)

    df_dt = grad(f, 2)
    d2f_dtx = grad(df_dt, 0)
    d2f_dty = grad(df_dt, 1)
    d2f_dt2 = grad(df_dt, 2)

    df_dx = vmap(df_dx, (0, 0, 0, None))
    df_dy = vmap(df_dy, (0, 0, 0, None))
    df_dt = vmap(df_dt, (0, 0, 0, None))

    d2f_dx2 = vmap(d2f_dx2, (0, 0, 0, None))
    d2f_dy2 = vmap(d2f_dy2, (0, 0, 0, None))
    d2f_dtx = vmap(d2f_dtx, (0, 0, 0, None))
    d2f_dty = vmap(d2f_dty, (0, 0, 0, None))
    d2f_dt2 = vmap(d2f_dt2, (0, 0, 0, None))

    return f, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty


def tuple_to_dict_parameters(
        model: nn.Module, params: tuple[torch.nn.Parameter, ...]
) -> OrderedDict[str, torch.nn.Parameter]:
    """Convert a set of parameters stored as a tuple into a dictionary form

    This conversion is required to be able to call the `functional_call` API which requires
    parameters in a dictionary form from the results of a functional optimization step which 
    returns the parameters as a tuple

    Args:
        model (nn.Module): the model to make the functional calls for. It can be any subclass of
            a nn.Module
        params (tuple[Parameter, ...]): the model parameters stored as a tuple
    
    Returns:
        An OrderedDict instance with the parameters stored as an ordered dictionary
    """
    keys = list(dict(model.named_parameters()).keys())
    values = list(params)
    return OrderedDict(({k:v for k,v in zip(keys, values)}))

def get_EM_values(x: torch.Tensor, y: torch.Tensor, geometry: torch.Tensor):
    """
    Returns the EM values of the point, given its coordinates and the geometry
    """
    percent_x = (x - SPATIAL_DOMAIN_SIZE[0][0]) / (SPATIAL_DOMAIN_SIZE[0][1] - SPATIAL_DOMAIN_SIZE[0][0])
    index_x = percent_x * (IMG_SIZE[0])
    percent_y = (y - SPATIAL_DOMAIN_SIZE[1][0]) / (SPATIAL_DOMAIN_SIZE[1][1] - SPATIAL_DOMAIN_SIZE[1][0])
    index_y = percent_y * (IMG_SIZE[1])

    index_x = torch.clamp(index_x, 0, IMG_SIZE[0] - 1)
    index_y = torch.clamp(index_x, 0, IMG_SIZE[1] - 1)

    return geometry[:, index_x.int(), index_y.int()]
 
def L4loss(pred: torch.Tensor, target: torch.Tensor):
    assert pred.shape == target.shape
    return ((pred - target)**4).mean()

def L6loss(pred: torch.Tensor, target: torch.Tensor):
    assert pred.shape == target.shape
    return ((pred - target)**6).mean()
    
def get_PINN_warmup_loss_fn(f, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty, training_points_loss_fn):
    def loss_fn(x, y, t, u,  
                boundary_points: torch.Tensor,
                collocation_points_xyt: torch.Tensor, 
                geometry: np.ndarray,
                params):
        """
        Loss function for the network:
        
        Parameters
        ----------
        `x`, `y`, and `t` are the inputs to the network, `u` is the output electric field.
        
        `domain_size` is the time and spatial size of the domain in shape [t, y, x], in
        where to compute the physics (collocation) loss.
        """

        # training points:
        train_preds = f(x, y, t, params)
        train_loss = training_points_loss_fn(train_preds, u)
        return train_loss, 0

    return loss_fn

def get_PINN_loss_fn(f, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty, training_points_loss_fn):
    def loss_fn(x, y, t, u,  
                boundary_points: torch.Tensor,
                collocation_points_xyt: torch.Tensor, 
                geometry: np.ndarray,
                params):
        """
        Loss function for the network:
        
        Parameters
        ----------
        `x`, `y`, and `t` are the inputs to the network, `u` is the output electric field.
        
        `domain_size` is the time and spatial size of the domain in shape [t, y, x], in
        where to compute the physics (collocation) loss.
        """

        # training points:
        train_preds = f(x, y, t, params)
        train_loss = training_points_loss_fn(train_preds, u)

        l = nn.MSELoss()
        # # boundary conditions:
        # boundary_points_xmin = boundary_points[0]
        # boundary_points_ymin = boundary_points[1]
        # boundary_points_xmax = boundary_points[2]
        # boundary_points_ymax = boundary_points[3]
        

        # free-surface: waves are free to propagate at the borders (or only upper y?)
        # lapl(f(x, y=0, t)) = 0
        # tb, yb, xb = boundary_points
        # boundary_loss = d2f_dx2(xb, yb, tb, params) + d2f_dy2(xb, yb, tb, params)
        # boundary_loss = l(boundary_loss, torch.zeros_like(boundary_loss))

        # paraxial absorbing conditions:
        # 2nd order:
        # u_tt - u_tx - 1/2 u_yy|x=0 = 0
        # u_tt - u_ty - 1/2 u_xx|y=0 = 0
        # u_tt + u_tx - 1/2 u_yy|x=a = 0
        # u_tt + u_ty - 1/2 u_xx|y=b = 0
        # xb, yb, tb = boundary_points_xmin
        # boundary_parax_loss_xmin = d2f_dt2(xb, yb, tb, params) - d2f_dtx(xb, yb, tb, params) - 0.5 * d2f_dy2(xb, yb, tb, params)
        # boundary_parax_loss_xmin = l(boundary_parax_loss_xmin, torch.zeros_like(boundary_parax_loss_xmin))

        # xb, yb, tb = boundary_points_ymin
        # boundary_parax_loss_ymin = d2f_dt2(xb, yb, tb, params) - d2f_dty(xb, yb, tb, params) - 0.5 * d2f_dx2(xb, yb, tb, params)
        # boundary_parax_loss_ymin = l(boundary_parax_loss_ymin, torch.zeros_like(boundary_parax_loss_ymin))

        # xb, yb, tb = boundary_points_xmax
        # boundary_parax_loss_xmax = d2f_dt2(xb, yb, tb, params) + d2f_dtx(xb, yb, tb, params) - 0.5 * d2f_dy2(xb, yb, tb, params)
        # boundary_parax_loss_xmax = l(boundary_parax_loss_xmax, torch.zeros_like(boundary_parax_loss_xmax))

        # xb, yb, tb = boundary_points_ymax
        # boundary_parax_loss_ymax = d2f_dt2(xb, yb, tb, params) + d2f_dty(xb, yb, tb, params) - 0.5 * d2f_dx2(xb, yb, tb, params)
        # boundary_parax_loss_ymax = l(boundary_parax_loss_ymax, torch.zeros_like(boundary_parax_loss_ymax))

        # boundary_parax_loss = boundary_parax_loss_xmin + boundary_parax_loss_ymin + boundary_parax_loss_xmax + boundary_parax_loss_ymax


        # collocation points:
        # d2f_dt2 - 1/(mu*eps) * (d2f_dx2 + d2f_dy2) + (sigma/epsilon)*df_dt = 0


        xc, yc, tc = collocation_points_xyt
        EM_values = get_EM_values(xc, yc, geometry)

        epsilon, sigma, mu, _ = EM_values

        epsilon *= EPSILON_0
        sigma = 0
        mu *= MU_0


        xc.requires_grad_()
        tc.requires_grad_()
        yc.requires_grad_()
        uc = f(xc, yc, tc, params)

        dfx = torch.autograd.grad(uc, xc, torch.ones_like(uc), create_graph=True)[0]
        dfy = torch.autograd.grad(uc, yc, torch.ones_like(uc), create_graph=True)[0]
        dft = torch.autograd.grad(uc, tc, torch.ones_like(uc), create_graph=True)[0]
        dftt = torch.autograd.grad(dft, tc, torch.ones_like(dft), create_graph=True)[0]
        dfxx = torch.autograd.grad(dfx, xc, torch.ones_like(dfx), create_graph=True)[0]
        dfyy = torch.autograd.grad(dfy, yc, torch.ones_like(dfy), create_graph=True)[0]

        # print("Collocation points:")
        # print("x:", xc[0])
        # print("y:", yc[0])
        # print("t:", tc[0])
        # print("u:", uc[0])

        # dftt = d2f_dt2(xc, yc, tc, params)
        # dfxx = d2f_dx2(xc, yc, tc, params)
        # dfyy = d2f_dy2(xc, yc, tc, params)
        # dft = df_dt(xc, yc, tc, params)
        # print("dftt:", dftt)
        # print("dfxx:", dfxx)
        # print("dfyy:", dfyy)
        # print("dft:", dft)


        term1 = dftt
        term2 = (1/(epsilon*mu)) * (dfxx + dfyy)
        term3 = dft * sigma / epsilon

        # print("term1:", term1)
        # print("term2:", term2)
        # print("term3:", term3)


        # collocation_loss = dftt - (1/(epsilon*mu)) * (dfxx + dfyy) + dft * sigma / epsilon
        collocation_loss = term1 - term2 + term3
        collocation_loss = l(2e-17 * collocation_loss, torch.zeros_like(collocation_loss))


        # physics_loss = l(physics_loss, torch.zeros_like(physics_loss)) + l(boundary_loss, torch.zeros_like(boundary_loss))
        physics_loss = collocation_loss
        # print("Physics loss:", physics_loss)

        # physics_loss = 0

        return train_loss, physics_loss

    return loss_fn

def predict(model: MLP, samples: torch.Tensor):
    model.eval()
    x, y, t, u = samples
    predictions = model(x, y, t).cpu().detach().squeeze()
    
    return predictions

def predict_functional(f,samples: torch.Tensor, params: tuple):
    predictions = []
    x, y, t, u = samples
    predictions = f(x, y, t, params).cpu().detach().squeeze()
    
    return predictions

def predict_frame(model: MLP, frame: np.ndarray) -> torch.Tensor:
    model = model.cpu()
    model.eval()
    predictions = torch.zeros((frame.shape[-2], frame.shape[-1]))
    for _y in range(frame.shape[-2]):
        for _x in range(frame.shape[-1]):
            x = frame[0][_y][_x]
            y = frame[1][_y][_x]
            t = frame[2][_y][_x]
            xt = torch.tensor(x, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.float32)
            tt = torch.tensor(t, dtype=torch.float32)
            u = model(xt, yt, tt).detach()
            predictions[_y, _x] = u
    
    return predictions

def evaluate_functional(f: Callable, samples: torch.Tensor, loss_fn, params):
    losses = []
    x, y, t, u = samples
    loss = loss_fn(f(x, y, t, params), u).cpu().detach()
    losses.append(loss)
    
    return float(torch.tensor(losses).mean())

def evaluate(model: MLP, samples: torch.Tensor, regular_loss_fn):
    model.eval()
    losses = []
    x, y, t, u = samples
    loss = regular_loss_fn(model(x, y, t), u.view(-1, 1)).cpu().detach()
    losses.append(loss)
    
    return float(torch.tensor(losses).mean())

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

def show_predictions(f_PINN: Callable, f_regular: Callable, PINN_params: tuple, regular_params: tuple, samples: torch.Tensor):
    # show predictions of the field for NN and PINN
    ground_truth = samples[3].reshape(IMG_SIZE).cpu()
    regular_predictions =  predict_functional(f_regular, samples, regular_params)
    regular_predictions = regular_predictions.reshape(IMG_SIZE)
    PINN_predictions =  predict_functional(f_PINN, samples, PINN_params)
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
    plt.show()

def plot_data_histogram(data: torch.Tensor, labels: torch.Tensor):

    labels = labels[torch.abs(labels) >= 1e-4]

    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0][0].hist(data[:, 0], bins=100)
    axs[0][0].set_title("x coordinate")
    axs[0][1].hist(data[:, 1], bins=100)
    axs[0][1].set_title("y coordinate")
    axs[1][0].hist(data[:, 2], bins=100)
    axs[1][0].set_title("t coordinate")
    axs[1][1].hist(labels, bins=100)
    axs[1][1].set_title("field values")
    plt.tight_layout()
    plt.show()

def train(PINN_model: MLP, 
          f_PINN: Callable,
          PINN_optimizer: torch.optim.Optimizer,
          PINN_loss_fn: Callable,
          regular_model: MLP,
          f_regular: Callable,
          regular_optimizer: MLP,
          regular_loss_fn: Callable,
          train_samples: torch.Tensor,
          val_samples: torch.Tensor,
          boundary_points: torch.Tensor,
          collocation_points: torch.Tensor,
          geometry: torch.Tensor,
          epochs: int):
    
    PINN_params_tuple = tuple(PINN_model.parameters())
    regular_params_tuple = tuple(regular_model.parameters())

    PINN_train_loss_evolution = []
    PINN_physics_loss_evolution = []
    regular_loss_evolution = []

    best_regular_model = None
    best_PINN_model = None

    PINN_val_loss_evolution = []
    regular_val_loss_evolution = []
    for e in tqdm(range(epochs)):
        # print("Epoch", i)
        PINN_model.train()
        regular_model.train()
        x, y, t, u  = train_samples
        PINN_optimizer.zero_grad()

        train_loss, physics_loss = PINN_loss_fn(x, y, t, u, boundary_points, collocation_points, geometry, PINN_params_tuple)
        PINN_loss = train_loss + physics_loss
        PINN_loss.backward()
        PINN_optimizer.step()
        PINN_train_loss_evolution.append(float(train_loss))
        PINN_physics_loss_evolution.append(float(physics_loss))

        # update the regular model
        regular_optimizer.zero_grad()
        regular_predictions = f_regular(x, y, t, regular_params_tuple)
        # _, regular_predictions = scaler.inverse_transform(None, regular_predictions)
        regular_loss = regular_loss_fn(regular_predictions, u)
        regular_loss.backward()
        regular_optimizer.step()
        regular_loss_evolution.append(float(regular_loss))

        # compute validation loss
        PINN_val_loss_evolution.append(evaluate_functional(f_PINN, val_samples, regular_loss_fn, PINN_params_tuple))
        regular_val_loss_evolution.append(evaluate_functional(f_regular, val_samples, regular_loss_fn, regular_params_tuple))
        if (e + 1) % LOGGING_FREQUENCY == 0:
            print(f"End of epoch {e}:")
            print(f"PINN train loss {PINN_train_loss_evolution[-1]}")
            print(f"PINN val loss {PINN_val_loss_evolution[-1]}")
            print(f"PINN physics loss {PINN_physics_loss_evolution[-1]}")
            print(f"NN train loss {regular_loss_evolution[-1]}")
            print(f"NN val loss {regular_val_loss_evolution[-1]}")

        # check best model
        if best_PINN_model is None or PINN_val_loss_evolution[-1] < best_PINN_model[0]:
            best_PINN_model = PINN_val_loss_evolution[-1], copy.deepcopy(PINN_model).cpu()
        if best_regular_model is None or regular_val_loss_evolution[-1] < best_regular_model[0]:
            best_regular_model = regular_val_loss_evolution[-1], copy.deepcopy(regular_model).cpu()

    fig, ax = plt.subplots()
    ax.semilogy(regular_loss_evolution, label="NN loss")
    ax.semilogy(PINN_train_loss_evolution, label="PINN train loss")
    ax.semilogy(PINN_physics_loss_evolution, label="PINN physics loss")
    ax.set(title="Train loss evolution", xlabel="# step", ylabel="Loss")
    ax.legend()

    plt.show()

    fig, ax = plt.subplots()
    ax.semilogy(regular_val_loss_evolution, label="NN loss")
    ax.semilogy(PINN_val_loss_evolution, label="PINN loss")
    ax.set(title="Validation loss evolution", xlabel="# epochs", ylabel="Loss")
    ax.legend()

    plt.show()

    return best_PINN_model[1], PINN_model, best_regular_model[1], regular_model


def uniform_material():
    snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]

    if IMG_SIZE != (200, 200):
        snaps = []
        for s in snapshots:
            s = cv2.resize(s, IMG_SIZE)
            snaps.append(s)
        snapshots = np.array(snaps)


    geometry = torch.broadcast_to(torch.tensor([1, 0, 1, 0]), (IMG_SIZE[0], IMG_SIZE[1], 4)).transpose(0, 2) 
    geometry = geometry.float().to(DEVICE)

    # define models and optimizers
    PINN_model = MLP(3, [64, 64, 64, 64, 64], 1, nn.SiLU)
    # PINN_model.load_state_dict(torch.load("checkpoints/NN_model_best_L4_20k_silu_5x256.ckp"), strict=True)
    # PINN_model.apply(PINN_model.init_weights)
    PINN_model = PINN_model.to(DEVICE)
    PINN_params_tuple = tuple(PINN_model.parameters())

    regular_model = MLP(3, [64, 64, 64, 64, 64], 1, nn.ReLU)
    # regular_model.load_state_dict(torch.load("checkpoints/NN_model_best_L4_20k_silu_5x256.ckp"), strict=True)
    regular_model = regular_model.to(DEVICE)
    regular_params_tuple = tuple(regular_model.parameters())


    PINN_optimizer = torchopt.Adam(PINN_params_tuple, lr = LR)
    regular_optimizer = torch.optim.Adam(regular_model.parameters(), lr = LR)


    # Create the dataset
    train_indexes = list(range(15, 30))
    train_dataset = PaperDataset(snapshots[train_indexes], t_offsets=train_indexes)
    print("Train dataset points:")
    print("Data shape:", train_dataset.data.shape)
    print("Data min:", train_dataset.data.min(dim=0).values)
    print("Data max:", train_dataset.data.max(dim=0).values)
    print("Label shape:", train_dataset.labels.shape)
    print("Label min:", train_dataset.labels.min(dim=0).values)
    print("Label max:", train_dataset.labels.max(dim=0).values)
    print()
    # save_field_animation(train_dataset.snapshots.reshape((-1, 200, 200)), None, interval=50, bound_mult_factor=0.0005)
    # frame_15ns = train_dataset.get_frame(1)
    #show_field(frame_15ns)
    scaler = train_dataset.scaler
    val_indexes = [25]
    val_dataset = PaperDataset(snapshots[val_indexes], t_offsets=val_indexes, scaler=scaler)
    print("Validation dataset points:")
    print("Data min:", val_dataset.data.min(dim=0).values)
    print("Data max:", val_dataset.data.max(dim=0).values)
    print("Label min:", val_dataset.labels.min(dim=0).values)
    print("Label max:", val_dataset.labels.max(dim=0).values)
    print()

    test_indexes = [40]
    test_dataset = PaperDataset(snapshots[test_indexes], t_offsets=test_indexes, scaler=scaler)
    print("Test dataset points:")
    print("Data min:", test_dataset.data.min(dim=0).values)
    print("Data max:", test_dataset.data.max(dim=0).values)
    print("Label min:", test_dataset.labels.min(dim=0).values)
    print("Label max:", test_dataset.labels.max(dim=0).values)
    print()
    # save_field_animation(val_dataset.snapshots.reshape((-1, 5, 5)), None, interval=50)
    # frame_60ns = val_dataset.get_frame(0)
    # show_field(frame_60ns)


    # collocation points
    collocation_points = RNG.uniform(size=(3, N_COLLOCATION_POINTS)) * np.array(COLLOCATION_DOMAIN_SIZE).reshape(3, -1)
    tc, yc, xc = collocation_points
    tc = tc + 1.5e-8
    collocation_points = np.stack([xc, yc, tc])
    collocation_points = torch.tensor(collocation_points, dtype=torch.float32)
    # collocation_points, _ = scaler.transform(collocation_points.T, None)
    collocation_points = collocation_points.to(DEVICE)

    print("collocation points:", collocation_points.shape)
    print("min:", collocation_points.min(axis=1).values)
    print("max:", collocation_points.max(axis=1).values)

    # boundary points
    # TODO: works only if x and y size are the same
    boundary_points = RNG.uniform(size=(2, N_BOUNDARY_POINTS * 4)) * np.array((COLLOCATION_DOMAIN_SIZE[1], COLLOCATION_DOMAIN_SIZE[0])).reshape(2, -1)
    print("bound shape:", boundary_points.shape)
    print("min:", boundary_points.min(axis=1))
    print("max:", boundary_points.max(axis=1))

    boundary_points_xmin = np.stack([np.zeros(N_BOUNDARY_POINTS), boundary_points[0, :N_BOUNDARY_POINTS], boundary_points[1, :N_BOUNDARY_POINTS]])
    boundary_points_ymin = np.stack([boundary_points[0, N_BOUNDARY_POINTS:2*N_BOUNDARY_POINTS], np.zeros(N_BOUNDARY_POINTS), boundary_points[1, N_BOUNDARY_POINTS:2*N_BOUNDARY_POINTS]])
    boundary_points_xmax = np.stack([np.full((N_BOUNDARY_POINTS), IMG_SIZE[1]), boundary_points[0, 2*N_BOUNDARY_POINTS:3*N_BOUNDARY_POINTS], boundary_points[1, 2*N_BOUNDARY_POINTS:3*N_BOUNDARY_POINTS]])
    boundary_points_ymax = np.stack([boundary_points[0, 3*N_BOUNDARY_POINTS:4*N_BOUNDARY_POINTS], np.full((N_BOUNDARY_POINTS), IMG_SIZE[0]), boundary_points[1, 3*N_BOUNDARY_POINTS:4*N_BOUNDARY_POINTS]])
    boundary_points = [boundary_points_xmin, boundary_points_ymin, boundary_points_xmax, boundary_points_ymax]
    boundary_points = np.stack(boundary_points)
    boundary_points = torch.tensor(boundary_points, dtype=torch.float32)
    boundary_points.to(DEVICE)

    print("boundary points:", boundary_points.shape)
    print("min:", boundary_points.min(axis=2).values)
    print("max:", boundary_points.max(axis=2).values)

    # get the derivative functions
    f_PINN, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty = get_f_and_derivatives(PINN_model, scaler)
    PINN_loss_fn_L4 = get_PINN_warmup_loss_fn(f_PINN, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty, L4loss)
    PINN_loss_fn_L2 = get_PINN_loss_fn(f_PINN, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty, nn.MSELoss())
    regular_loss_fn = nn.MSELoss()

    # get f for regular model
    f_regular = get_f_and_derivatives(regular_model, scaler)[0]
    #plot_data_histogram(train_dataset.data, train_dataset.labels)

    input()
    # epsilon = geometry[0, y_indeces, x_indeces]
    # sigma = geometry[1, y_indeces, x_indeces]
    # mu = geometry[2, y_indeces, x_indeces]

    # plt.show()

    # generate dataset
    print("building train dataset...")
    train_samples = torch.cat([train_dataset.data, train_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    print("building validation dataset...")
    val_samples = torch.cat([val_dataset.data, val_dataset.labels[:, None]], dim=1).T.to(DEVICE)


    print("building test dataset...")
    test_samples = torch.cat([test_dataset.data, test_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train(PINN_model,
                                                f_PINN,
                                                PINN_optimizer,
                                                PINN_loss_fn_L4,
                                                regular_model,
                                                f_regular,
                                                regular_optimizer,
                                                L4loss,
                                                train_samples,
                                                val_samples,
                                                boundary_points,
                                                collocation_points,
                                                geometry,
                                                EPOCHS_WARMUP)

    best_PINN_model = best_PINN_model.to(DEVICE)
    best_regular_model = best_regular_model.to(DEVICE)
    regular_optimizer = torch.optim.Adam(best_regular_model.parameters(), lr = LR)

    f_PINN = get_f_and_derivatives(best_PINN_model, scaler)[0]
    f_regular = get_f_and_derivatives(best_regular_model, scaler)[0]
    PINN_optimizer = torchopt.Adam(tuple(best_PINN_model.parameters()), lr = LR)

    best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train(best_PINN_model,
                                                f_PINN,
                                                PINN_optimizer,
                                                PINN_loss_fn_L2,
                                                best_regular_model,
                                                f_regular,
                                                regular_optimizer,
                                                regular_loss_fn,
                                                train_samples,
                                                val_samples,
                                                boundary_points,
                                                collocation_points,
                                                geometry,
                                                EPOCHS)

    torch.save(last_PINN_model.state_dict(), "checkpoints/PINN_model_last.ckp")
    torch.save(best_PINN_model.state_dict(), "checkpoints/PINN_model_best.ckp")
    torch.save(last_regular_model.state_dict(), "checkpoints/NN_model_last.ckp")
    torch.save(best_regular_model.state_dict(), "checkpoints/NN_model_best.ckp")

    PINN_params_tuple = tuple(best_PINN_model.parameters())
    regular_params_tuple = tuple(best_regular_model.parameters())
    best_PINN_model = best_PINN_model.to(DEVICE)
    best_regular_model = best_regular_model.to(DEVICE)

    show_predictions(f_PINN, f_regular, PINN_params_tuple, regular_params_tuple, val_samples)
    show_predictions(f_PINN, f_regular, PINN_params_tuple, regular_params_tuple, test_samples)

def two_layer():
    snapshots = np.load("paper_data/2layer_wavefield.npz")["00000_E"]

    if IMG_SIZE != (200, 200):
        snaps = []
        for s in snapshots:
            s = cv2.resize(s, IMG_SIZE)
            snaps.append(s)
        snapshots = np.array(snaps)


    geometry = np.load("paper_data/2layer_geometry.npy")
    l = []
    for img in geometry:
        i = cv2.resize(img, IMG_SIZE)
        l.append(i)
    
    geometry = np.asarray(l)

    plt.imshow(geometry[0])
    plt.show()

    geometry = torch.from_numpy(np.array(l)).to(DEVICE)

    values = get_EM_values(torch.Tensor([1, 10, 19]), torch.tensor([1, 10, 19]), geometry)
    print(values.shape)


    # define models and optimizers
    PINN_model = MLP(3, [64, 64, 64, 64, 64], 1, nn.SiLU)
    # PINN_model.load_state_dict(torch.load("checkpoints/NN_model_best_L4_20k_silu_5x256.ckp"), strict=True)
    # PINN_model.apply(PINN_model.init_weights)
    PINN_model = PINN_model.to(DEVICE)
    PINN_params_tuple = tuple(PINN_model.parameters())

    regular_model = MLP(3, [64, 64, 64, 64, 64], 1, nn.SiLU)
    # regular_model.load_state_dict(torch.load("checkpoints/NN_model_best_L4_20k_silu_5x256.ckp"), strict=True)
    regular_model = regular_model.to(DEVICE)
    regular_params_tuple = tuple(regular_model.parameters())


    PINN_optimizer = torchopt.Adam(PINN_params_tuple, lr = LR)
    regular_optimizer = torch.optim.Adam(regular_model.parameters(), lr = LR)


    # Create the dataset
    train_indexes = list(range(15, 40))
    train_dataset = PaperDataset(snapshots[train_indexes], t_offsets=train_indexes)
    print("Train dataset points:")
    print("Data shape:", train_dataset.data.shape)
    print("Data min:", train_dataset.data.min(dim=0).values)
    print("Data max:", train_dataset.data.max(dim=0).values)
    print("Label shape:", train_dataset.labels.shape)
    print("Label min:", train_dataset.labels.min(dim=0).values)
    print("Label max:", train_dataset.labels.max(dim=0).values)
    print()
    save_field_animation(train_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=50)
    # frame_15ns = train_dataset.get_frame(1)
    #show_field(frame_15ns)
    scaler = train_dataset.scaler
    val_indexes = [25]
    val_dataset = PaperDataset(snapshots[val_indexes], t_offsets=val_indexes, scaler=scaler)
    print("Validation dataset points:")
    print("Data min:", val_dataset.data.min(dim=0).values)
    print("Data max:", val_dataset.data.max(dim=0).values)
    print("Label min:", val_dataset.labels.min(dim=0).values)
    print("Label max:", val_dataset.labels.max(dim=0).values)
    print()

    test_indexes = [50]
    test_dataset = PaperDataset(snapshots[test_indexes], t_offsets=test_indexes, scaler=scaler)
    print("Test dataset points:")
    print("Data min:", test_dataset.data.min(dim=0).values)
    print("Data max:", test_dataset.data.max(dim=0).values)
    print("Label min:", test_dataset.labels.min(dim=0).values)
    print("Label max:", test_dataset.labels.max(dim=0).values)
    print()
    save_field_animation(test_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=50)
    # frame_60ns = val_dataset.get_frame(0)
    # show_field(frame_60ns)


    # collocation points
    collocation_points = RNG.uniform(size=(3, N_COLLOCATION_POINTS)) * np.array(COLLOCATION_DOMAIN_SIZE).reshape(3, -1)
    tc, yc, xc = collocation_points
    tc = tc + 1.5e-8
    collocation_points = np.stack([xc, yc, tc])
    collocation_points = torch.tensor(collocation_points, dtype=torch.float32)
    # collocation_points, _ = scaler.transform(collocation_points.T, None)
    collocation_points = collocation_points.to(DEVICE)

    print("collocation points:", collocation_points.shape)
    print("min:", collocation_points.min(axis=1).values)
    print("max:", collocation_points.max(axis=1).values)

    # boundary points
    # TODO: works only if x and y size are the same
    boundary_points = RNG.uniform(size=(2, N_BOUNDARY_POINTS * 4)) * np.array((COLLOCATION_DOMAIN_SIZE[1], COLLOCATION_DOMAIN_SIZE[0])).reshape(2, -1)
    print("bound shape:", boundary_points.shape)
    print("min:", boundary_points.min(axis=1))
    print("max:", boundary_points.max(axis=1))

    boundary_points_xmin = np.stack([np.zeros(N_BOUNDARY_POINTS), boundary_points[0, :N_BOUNDARY_POINTS], boundary_points[1, :N_BOUNDARY_POINTS]])
    boundary_points_ymin = np.stack([boundary_points[0, N_BOUNDARY_POINTS:2*N_BOUNDARY_POINTS], np.zeros(N_BOUNDARY_POINTS), boundary_points[1, N_BOUNDARY_POINTS:2*N_BOUNDARY_POINTS]])
    boundary_points_xmax = np.stack([np.full((N_BOUNDARY_POINTS), IMG_SIZE[1]), boundary_points[0, 2*N_BOUNDARY_POINTS:3*N_BOUNDARY_POINTS], boundary_points[1, 2*N_BOUNDARY_POINTS:3*N_BOUNDARY_POINTS]])
    boundary_points_ymax = np.stack([boundary_points[0, 3*N_BOUNDARY_POINTS:4*N_BOUNDARY_POINTS], np.full((N_BOUNDARY_POINTS), IMG_SIZE[0]), boundary_points[1, 3*N_BOUNDARY_POINTS:4*N_BOUNDARY_POINTS]])
    boundary_points = [boundary_points_xmin, boundary_points_ymin, boundary_points_xmax, boundary_points_ymax]
    boundary_points = np.stack(boundary_points)
    boundary_points = torch.tensor(boundary_points, dtype=torch.float32)
    boundary_points.to(DEVICE)

    print("boundary points:", boundary_points.shape)
    print("min:", boundary_points.min(axis=2).values)
    print("max:", boundary_points.max(axis=2).values)

    # get the derivative functions
    f_PINN, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty = get_f_and_derivatives(PINN_model, scaler)
    PINN_loss_fn_L4 = get_PINN_warmup_loss_fn(f_PINN, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty, L4loss)
    PINN_loss_fn_L2 = get_PINN_loss_fn(f_PINN, df_dx, df_dy, df_dt, d2f_dx2, d2f_dy2, d2f_dt2, d2f_dtx, d2f_dty, nn.MSELoss())
    regular_loss_fn = nn.MSELoss()

    # get f for regular model
    f_regular = get_f_and_derivatives(regular_model, scaler)[0]
    #plot_data_histogram(train_dataset.data, train_dataset.labels)

    input()
    # epsilon = geometry[0, y_indeces, x_indeces]
    # sigma = geometry[1, y_indeces, x_indeces]
    # mu = geometry[2, y_indeces, x_indeces]

    # plt.show()

    # generate dataset
    print("building train dataset...")
    train_samples = torch.cat([train_dataset.data, train_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    print("building validation dataset...")
    val_samples = torch.cat([val_dataset.data, val_dataset.labels[:, None]], dim=1).T.to(DEVICE)


    print("building test dataset...")
    test_samples = torch.cat([test_dataset.data, test_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    # best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train(PINN_model,
    #                                             f_PINN,
    #                                             PINN_optimizer,
    #                                             PINN_loss_fn_L4,
    #                                             regular_model,
    #                                             f_regular,
    #                                             regular_optimizer,
    #                                             L4loss,
    #                                             train_samples,
    #                                             val_samples,
    #                                             boundary_points,
    #                                             collocation_points,
    #                                             geometry,
    #                                             EPOCHS_WARMUP)

    # best_PINN_model = best_PINN_model.to(DEVICE)
    # best_regular_model = best_regular_model.to(DEVICE)
    # regular_optimizer = torch.optim.Adam(best_regular_model.parameters(), lr = LR)

    # f_PINN = get_f_and_derivatives(best_PINN_model, scaler)[0]
    # f_regular = get_f_and_derivatives(best_regular_model, scaler)[0]
    # PINN_optimizer = torchopt.Adam(tuple(best_PINN_model.parameters()), lr = LR)

    best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train(PINN_model,
                                                f_PINN,
                                                PINN_optimizer,
                                                PINN_loss_fn_L2,
                                                regular_model,
                                                f_regular,
                                                regular_optimizer,
                                                regular_loss_fn,
                                                train_samples,
                                                val_samples,
                                                boundary_points,
                                                collocation_points,
                                                geometry,
                                                EPOCHS)

    torch.save(last_PINN_model.state_dict(), "checkpoints/PINN_model_last.ckp")
    torch.save(best_PINN_model.state_dict(), "checkpoints/PINN_model_best.ckp")
    torch.save(last_regular_model.state_dict(), "checkpoints/NN_model_last.ckp")
    torch.save(best_regular_model.state_dict(), "checkpoints/NN_model_best.ckp")

    PINN_params_tuple = tuple(best_PINN_model.parameters())
    regular_params_tuple = tuple(best_regular_model.parameters())
    best_PINN_model = best_PINN_model.to(DEVICE)
    best_regular_model = best_regular_model.to(DEVICE)

    show_predictions(f_PINN, f_regular, PINN_params_tuple, regular_params_tuple, val_samples)
    show_predictions(f_PINN, f_regular, PINN_params_tuple, regular_params_tuple, test_samples)

if __name__ == "__main__":
    # uniform_material()
    two_layer()