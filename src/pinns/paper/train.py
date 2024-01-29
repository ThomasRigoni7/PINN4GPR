from tqdm import tqdm
import copy
from typing import Callable

import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from src.pinns.paper.model import MLP, evaluate_functional

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
          geometry: torch.Tensor,
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
        x, y, t, u  = train_samples
        PINN_optimizer.zero_grad()

        train_loss, physics_loss = PINN_loss_fn(f_PINN, x, y, t, u, boundary_points, collocation_points, geometry)
        PINN_loss = train_loss + physics_loss
        PINN_loss.backward()
        PINN_optimizer.step()
        PINN_train_loss_evolution.append(float(train_loss))
        PINN_physics_loss_evolution.append(float(physics_loss))

        # update the regular model
        regular_optimizer.zero_grad()
        regular_predictions = f_regular(x, y, t)
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