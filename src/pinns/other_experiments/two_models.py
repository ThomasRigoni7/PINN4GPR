"""
Trains a model able to recognize which geometry to use, the MLP model has an additional input which specifies the geometry.
"""


import math
from typing import Callable
import copy
from tqdm import tqdm

import numpy as np
import torch
torch.manual_seed(42)
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import cv2


import matplotlib.pyplot as plt
from src.visualization.misc import save_field_animation
from src.pinns.paper.dataset import MyNormalizer
from src.pinns.paper.model import IMG_SIZE, get_EM_values, show_field

EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6

class MultipleGeometryDataset(Dataset):
    def __init__(self, snapshots: list[np.ndarray] | np.ndarray, geometries: list[np.ndarray], t_offsets: list[float], scaler: MyNormalizer = None):
        # input snapshots have shape [g, t, y, x] -> gets flattened in: iterate first x -> y -> t -> g

        if isinstance(snapshots, list):
            snapshots = np.asarray(snapshots)

        self.geometries = geometries
        self.snapshots_shape = snapshots.shape
        self.snapshots = snapshots.flatten()
        self.t_offsets = t_offsets
        data = []
        labels = []
        for index in range(len(self.snapshots)):
            x = index % self.snapshots_shape[3] / 10
            y = (index // self.snapshots_shape[2]) % self.snapshots_shape[2] / 10
            t = self.t_offsets[(index // (self.snapshots_shape[2] * self.snapshots_shape[3])) % self.snapshots_shape[1]] * 1e-9
            g = index // (self.snapshots_shape[1] * self.snapshots_shape[2] * self.snapshots_shape[3])
            u = self.snapshots[index]
            data.append((x, y, t, g))
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
        return *d, self.labels[index]
    
    def get_frame(self, geometry_index: int, frame_index: int):
        """
        Returns all the points associated with a specific frame

        Parameters
        ----------
        frame_index : int
            index of the frame to return
        
        Returns
        -------
        Tensor
            all the points related to the frame, in image shape of [height, width]
        """
        start_index = self.snapshots_shape[-1] * self.snapshots_shape[-2] * frame_index + \
            self.snapshots_shape[-3] * self.snapshots_shape[-1] * self.snapshots_shape[-2] * geometry_index
        points = []
        for i in range(start_index, start_index + self.snapshots_shape[-1] * self.snapshots_shape[-2]):
            p = self[i]
            points.append(p)
        points = np.array(points).transpose(1, 0)
        points = points.reshape(4, self.snapshots_shape[-2], self.snapshots_shape[-1])
        return points
    
    def print_info(self):
        print("Data shape:", self.data.shape)
        print("Data min:", self.data.min(dim=0).values)
        print("Data max:", self.data.max(dim=0).values)
        print("Label shape:", self.labels.shape)
        print("Label min:", self.labels.min(dim=0).values)
        print("Label max:", self.labels.max(dim=0).values)
        print()

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

    def forward(self, x, y, t, g):
        x = torch.stack([x, y, t, g])
        if x.ndim == 1:
            x = x[:, None]
        x = x.transpose(1, 0)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x

def get_f(model: MLP, scaler:MyNormalizer) -> Callable:
    def f(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:

        data = torch.stack([x, y, t, g])

        (x, y, t, g), _ = scaler.transform(data, None)

        u : torch.Tensor = model(x, y, t, g)
        u.squeeze()

        _, u = scaler.inverse_transform(None, u)
        return u.squeeze()

    return f

def get_PINN_loss_fn(training_points_loss_fn):
    def loss_fn(f, x, y, t, g, u,  
                boundary_points: torch.Tensor,
                collocation_points_xyt: torch.Tensor, 
                geometries: list[torch.Tensor]):
        """
        Loss function for the network:
        
        Parameters
        ----------
        `x`, `y`, and `t` are the inputs to the network, `u` is the output electric field.
        
        `domain_size` is the time and spatial size of the domain in shape [t, y, x], in
        where to compute the physics (collocation) loss.
        """

        # training points:
        train_preds = f(x, y, t, g)
        train_loss = training_points_loss_fn(train_preds, u)

        # collocation points
        l = nn.MSELoss()
        xc, yc, tc, gc = collocation_points_xyt
        EM_values_0 = get_EM_values(xc, yc, geometries[0])
        EM_values_1 = get_EM_values(xc, yc, geometries[1])

        EM_values = torch.where(gc==0, EM_values_0, EM_values_1)

        epsilon, sigma, mu, _ = EM_values
        epsilon *= EPSILON_0
        mu *= MU_0

        xc.requires_grad_()
        tc.requires_grad_()
        yc.requires_grad_()
        uc = f(xc, yc, tc, gc)

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

def predict_functional(f, samples: torch.Tensor):
    x, y, t, g, u = samples
    with torch.no_grad():
        predictions = f(x, y, t, g).cpu().detach().squeeze()
    
    return predictions

def show_predictions(f_PINN: Callable, f_regular: Callable, samples: torch.Tensor):
    # show predictions of the field for NN and PINN
    ground_truth = samples[-1].reshape(2, *IMG_SIZE).cpu()
    regular_predictions =  predict_functional(f_regular, samples)
    regular_predictions = regular_predictions.reshape(2, *IMG_SIZE)
    PINN_predictions =  predict_functional(f_PINN, samples)
    PINN_predictions = PINN_predictions.reshape(2, *IMG_SIZE)

    for gt, nn_p, pinn_p  in zip(ground_truth, regular_predictions, PINN_predictions):
        fig, axs = plt.subplots(nrows=2, ncols=3)
        mappable, vmin, vmax = show_field(gt, axs[0][1])
        axs[0][1].set_title("ground truth")
        show_field(nn_p, axs[0][0], vmin, vmax)
        axs[0][0].set_title("NN predictions")
        show_field(pinn_p, axs[0][2], vmin, vmax)
        axs[0][2].set_title("PINN_predictions")
        show_field(nn_p - gt, axs[1][0], vmin, vmax)
        axs[1][0].set_title("NN - GT")
        show_field(nn_p - pinn_p, axs[1][1], vmin, vmax)
        axs[1][1].set_title("NN - PINN")
        show_field(pinn_p - gt, axs[1][2], vmin, vmax)
        axs[1][2].set_title("PINN - GT")
        fig.colorbar(mappable, ax=axs, location="right", shrink=0.7)
        plt.show()

def evaluate_functional(f: Callable, samples: torch.Tensor, loss_fn):
    x, y, t, g, u = samples
    with torch.no_grad():
        loss = loss_fn(f(x, y, t, g), u).cpu().detach()
    
    return float(loss.mean())

LOGGING_FREQUENCY = 100
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
          geometries: list[torch.Tensor],
          epochs: int,
          use_scheduler:bool = False):
    
    if use_scheduler:
        PINN_scheduler = StepLR(PINN_optimizer, 1000, 0.9)
        regular_scheduler = StepLR(regular_optimizer, 1000, 0.9)


    fig, ax = plt.subplots()
    plt.ion()


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
        x, y, t, g, u  = train_samples
        PINN_optimizer.zero_grad()

        train_loss, physics_loss = PINN_loss_fn(f_PINN, x, y, t, g, u, boundary_points, collocation_points, geometries)
        PINN_loss = train_loss + physics_loss
        PINN_loss.backward()
        PINN_optimizer.step()
        PINN_train_loss_evolution.append(float(train_loss))
        PINN_physics_loss_evolution.append(float(physics_loss))

        # update the regular model
        regular_optimizer.zero_grad()
        regular_predictions = f_regular(x, y, t, g)
        # _, regular_predictions = scaler.inverse_transform(None, regular_predictions)
        regular_loss = regular_loss_fn(regular_predictions, u)
        regular_loss.backward()
        regular_optimizer.step()
        regular_loss_evolution.append(float(regular_loss))

        if use_scheduler:
            PINN_scheduler.step()
            regular_scheduler.step()

        # compute validation loss
        PINN_val_loss_evolution.append(evaluate_functional(f_PINN, val_samples, regular_loss_fn))
        regular_val_loss_evolution.append(evaluate_functional(f_regular, val_samples, regular_loss_fn))
        if (e + 1) % LOGGING_FREQUENCY == 0:
            print(f"End of epoch {e}:")
            print(f"PINN train loss {PINN_train_loss_evolution[-1]}")
            print(f"PINN val loss {PINN_val_loss_evolution[-1]}")
            print(f"PINN physics loss {PINN_physics_loss_evolution[-1]}")
            print(f"NN train loss {regular_loss_evolution[-1]}")
            print(f"NN val loss {regular_val_loss_evolution[-1]}")

            # draw
            ax.clear()
            ax.set(title="Train loss evolution", xlabel="# step", ylabel="Loss")
            ax.semilogy(regular_loss_evolution, label="NN loss")
            ax.semilogy(PINN_train_loss_evolution, label="PINN train loss")
            ax.semilogy(PINN_physics_loss_evolution, label="PINN physics loss")
            ax.legend()
            plt.show()
            plt.pause(0.01)

        # check best model
        if best_PINN_model is None or PINN_val_loss_evolution[-1] < best_PINN_model[0]:
            best_PINN_model = PINN_val_loss_evolution[-1], copy.deepcopy(PINN_model).cpu()
        if best_regular_model is None or regular_val_loss_evolution[-1] < best_regular_model[0]:
            best_regular_model = regular_val_loss_evolution[-1], copy.deepcopy(regular_model).cpu()


    plt.ioff()

    fig, ax = plt.subplots()
    ax.semilogy(regular_val_loss_evolution, label="NN loss")
    ax.semilogy(PINN_val_loss_evolution, label="PINN loss")
    ax.set(title="Validation loss evolution", xlabel="# epochs", ylabel="Loss")
    ax.legend()

    plt.show()

    return best_PINN_model[1], PINN_model, best_regular_model[1], regular_model

def two_model():
    """
    Trains a PINN model capable of distinguishing two geometries and NN and compares them.
    """

    DEVICE = "cuda:2"
    LR = 0.001
    RNG = np.random.default_rng(42)
    N_COLLOCATION_POINTS = 40000
    COLLOCATION_DOMAIN_SIZE = (30e-9, 20, 20)
    EPOCHS = 10000

    snapshots_uniform = np.load("paper_data/uniform_wavefield.npz")["0000_E"]
    snapshots_2layer = np.load("paper_data/2layer_wavefield.npz")["00000_E"]

    geometry_uniform = torch.broadcast_to(torch.tensor([1, 0, 1, 0]), (IMG_SIZE[0], IMG_SIZE[1], 4)).transpose(0, 2) 
    geometry_uniform = geometry_uniform.float().to(DEVICE)

    geometry_2layer = np.load("paper_data/2layer_geometry.npy")
    l = []
    for img in geometry_2layer:
        i = cv2.resize(img, IMG_SIZE)
        l.append(i)
    geometry_2layer = np.asarray(l)
    geometry_2layer = torch.from_numpy(np.array(l)).to(DEVICE)

    geometries = [geometry_uniform, geometry_2layer]

    # define models and optimizers
    PINN_model = MLP(4, [128, 128, 128, 128, 128], 1, nn.SiLU)
    PINN_model.load_state_dict(torch.load("checkpoints/PINN_two_model_20k.ckp"), strict=True)
    PINN_model = PINN_model.to(DEVICE)

    regular_model = MLP(4, [128, 128, 128, 128, 128], 1, nn.ReLU)
    regular_model.load_state_dict(torch.load("checkpoints/NN_two_model_20k.ckp"), strict=True)
    regular_model = regular_model.to(DEVICE)

    PINN_optimizer = torch.optim.Adam(PINN_model.parameters(), lr = LR)
    regular_optimizer = torch.optim.Adam(regular_model.parameters(), lr = LR)


    # Create the dataset
    train_indexes = list(range(15, 25))
    train_dataset = MultipleGeometryDataset([snapshots_uniform[train_indexes], snapshots_2layer[train_indexes]], [geometry_uniform, geometry_2layer], t_offsets=train_indexes)
    print("Train dataset points:")
    train_dataset.print_info()
    train_dataset.scaler.to(DEVICE)
    scaler = train_dataset.scaler
    save_field_animation(train_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=200)

    val_indexes = [25]
    val_dataset = MultipleGeometryDataset([snapshots_uniform[val_indexes], snapshots_2layer[val_indexes]], [geometry_uniform, geometry_2layer], t_offsets=val_indexes, scaler=scaler)
    print("Validation dataset points:")
    val_dataset.print_info()
    save_field_animation(val_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=200)

    test_indexes = [40]
    test_dataset = MultipleGeometryDataset([snapshots_uniform[test_indexes], snapshots_2layer[test_indexes]], [geometry_uniform, geometry_2layer], t_offsets=test_indexes, scaler=scaler)
    print("Test dataset points:")
    test_dataset.print_info()
    save_field_animation(test_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=200)


    # collocation points
    collocation_points = RNG.uniform(size=(3, N_COLLOCATION_POINTS)) * np.array(COLLOCATION_DOMAIN_SIZE).reshape(3, -1)
    gc = RNG.uniform(size=(N_COLLOCATION_POINTS)) > 0.5
    tc, yc, xc = collocation_points
    tc = tc + train_indexes[0] * 1e-9
    collocation_points = np.stack([xc, yc, tc, gc])
    collocation_points = torch.tensor(collocation_points, dtype=torch.float32)
    collocation_points = collocation_points.to(DEVICE)
    print("collocation points:", collocation_points.shape)
    print("min:", collocation_points.min(axis=1).values)
    print("max:", collocation_points.max(axis=1).values)

    # get the derivative functions
    f_PINN = get_f(PINN_model, scaler)
    PINN_loss_fn_L2 = get_PINN_loss_fn(nn.MSELoss())
    regular_loss_fn = nn.MSELoss()

    # get f for regular model
    f_regular = get_f(regular_model, scaler)
    #plot_data_histogram(train_dataset.data, train_dataset.labels)

    # generate dataset
    print("building train dataset...")
    train_samples = torch.cat([train_dataset.data, train_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    print("building validation dataset...")
    val_samples = torch.cat([val_dataset.data, val_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    print("building test dataset...")
    test_samples = torch.cat([test_dataset.data, test_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    show_predictions(f_PINN, f_regular, val_samples)
    show_predictions(f_PINN, f_regular, test_samples)

    exit()

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
                                                None,
                                                collocation_points,
                                                geometries,
                                                EPOCHS,
                                                use_scheduler=True)

    torch.save(last_PINN_model.state_dict(), "checkpoints/PINN_model_last.ckp")
    torch.save(best_PINN_model.state_dict(), "checkpoints/PINN_model_best.ckp")
    torch.save(last_regular_model.state_dict(), "checkpoints/NN_model_last.ckp")
    torch.save(best_regular_model.state_dict(), "checkpoints/NN_model_best.ckp")

    best_PINN_model = best_PINN_model.to(DEVICE)
    best_regular_model = best_regular_model.to(DEVICE)

    show_predictions(f_PINN, f_regular, val_samples)
    show_predictions(f_PINN, f_regular, test_samples)

if __name__ == "__main__":
    two_model()