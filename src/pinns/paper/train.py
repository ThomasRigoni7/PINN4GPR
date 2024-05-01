"""
This module contains functions to train both full-data and batched models,
in particular both a black box and PINN model are trained together on the same training data.
"""

from tqdm import tqdm
import copy
from typing import Callable
from pathlib import Path

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.pinns.paper.model import MLP, evaluate_functional

LOGGING_FREQUENCY_BATCHED = 1
LOGGING_FREQUENCY_FULL = 100

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
          use_scheduler:bool = False,
          results_folder: str | Path = None,
          interactive: bool = True):
    
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
        if (e + 1) % LOGGING_FREQUENCY_FULL == 0:
            # draw
            ax.clear()
            ax.set(title="Train loss evolution", xlabel="# step", ylabel="Loss")
            ax.semilogy(regular_loss_evolution, label="NN loss")
            ax.semilogy(PINN_train_loss_evolution, label="PINN train loss")
            ax.semilogy(PINN_physics_loss_evolution, label="PINN physics loss")
            ax.legend()
            if interactive:
                plt.show()
                plt.pause(0.01)

        # check best model
        if best_PINN_model is None or PINN_val_loss_evolution[-1] < best_PINN_model[0]:
            best_PINN_model = PINN_val_loss_evolution[-1], copy.deepcopy(PINN_model).cpu()
        if best_regular_model is None or regular_val_loss_evolution[-1] < best_regular_model[0]:
            best_regular_model = regular_val_loss_evolution[-1], copy.deepcopy(regular_model).cpu()


    plt.ioff()

    fig2, ax2 = plt.subplots()
    ax2.semilogy(regular_val_loss_evolution, label="NN loss")
    ax2.semilogy(PINN_val_loss_evolution, label="PINN loss")
    ax2.set(title="Validation loss evolution", xlabel="# epochs", ylabel="Loss")
    ax2.legend()

    if interactive:
        plt.show()

    if results_folder is not None:
        ax.clear()
        ax.set(title="Train loss evolution", xlabel="# step", ylabel="Loss")
        ax.semilogy(regular_loss_evolution, label="NN loss")
        ax.semilogy(PINN_train_loss_evolution, label="PINN train loss")
        ax.semilogy(PINN_physics_loss_evolution, label="PINN physics loss")
        ax.legend()
        fig.savefig(results_folder / "train_loss.png")
        fig2.savefig(results_folder / "val loss.png")

    return best_PINN_model[1], PINN_model, best_regular_model[1], regular_model



def train_batched(PINN_model: MLP, 
          f_PINN: Callable,
          PINN_optimizer: torch.optim.Optimizer,
          PINN_loss_fn: Callable,
          regular_model: MLP,
          f_regular: Callable,
          regular_optimizer: MLP,
          regular_loss_fn: Callable,
          train_loader: DataLoader,
          val_samples: torch.Tensor,
          boundary_points: torch.Tensor,
          collocation_points: torch.Tensor,
          geometry: torch.Tensor,
          epochs: int,
          device: str,
          use_scheduler:bool = False,
          results_folder: str | Path = None,
          interactive: bool = True):
    
    if use_scheduler:
        PINN_scheduler = StepLR(PINN_optimizer, 1000, 0.9)
        regular_scheduler = StepLR(regular_optimizer, 1000, 0.9)


    fig, ax = plt.subplots()
    if interactive:
        plt.ion()


    PINN_train_loss_step = []
    PINN_physics_loss_step = []
    NN_train_loss_step = []

    PINN_train_loss_epoch = []
    PINN_physics_loss_epoch = []
    NN_train_loss_epoch = []

    best_regular_model = None
    best_PINN_model = None

    PINN_val_loss_evolution = []
    regular_val_loss_evolution = []
    for e in tqdm(range(epochs), position=0, desc="Epoch"):
        for batch in tqdm(train_loader, position=1, desc="Step", leave=None):
            PINN_model.train()
            regular_model.train()
            PINN_optimizer.zero_grad()

            x, y, t, u  = batch
            x = x.to(device)
            y = y.to(device)
            t = t.to(device)
            u = u.to(device)

            train_loss, physics_loss = PINN_loss_fn(f_PINN, x, y, t, u, boundary_points, collocation_points, geometry)
            PINN_loss = train_loss + physics_loss
            PINN_loss.backward()
            PINN_optimizer.step()
            PINN_train_loss_step.append(float(train_loss))
            PINN_physics_loss_step.append(float(physics_loss))

            # update the regular model
            regular_optimizer.zero_grad()
            regular_predictions = f_regular(x, y, t)
            # _, regular_predictions = scaler.inverse_transform(None, regular_predictions)
            regular_loss = regular_loss_fn(regular_predictions, u)
            regular_loss.backward()
            regular_optimizer.step()
            NN_train_loss_step.append(float(regular_loss))

            if use_scheduler:
                PINN_scheduler.step()
                regular_scheduler.step()

        # compute validation loss
        PINN_val_loss_evolution.append(evaluate_functional(f_PINN, val_samples, regular_loss_fn))
        regular_val_loss_evolution.append(evaluate_functional(f_regular, val_samples, regular_loss_fn))

        # calculate the per epoch losses
        PINN_train_loss_epoch.append(float(torch.tensor([PINN_train_loss_step]).mean()))
        PINN_physics_loss_epoch.append(float(torch.tensor([PINN_physics_loss_step]).mean()))
        NN_train_loss_epoch.append(float(torch.tensor([NN_train_loss_step]).mean()))

        PINN_train_loss_step = []
        PINN_physics_loss_step = []
        NN_train_loss_step = []

        # draw
        ax.clear()
        ax.set(title="Train loss evolution", xlabel="# epoch", ylabel="Loss")
        ax.semilogy(NN_train_loss_epoch, label="NN loss")
        ax.semilogy(PINN_train_loss_epoch, label="PINN train loss")
        ax.semilogy(PINN_physics_loss_epoch, label="PINN physics loss")
        ax.semilogy(PINN_val_loss_evolution, label="PINN val loss")
        ax.semilogy(regular_val_loss_evolution, label="NN val loss")
        ax.set_ylim(1, None)
        ax.legend()
        if interactive:
            plt.show()
            plt.pause(0.01)

        # check best model
        if best_PINN_model is None or PINN_val_loss_evolution[-1] < best_PINN_model[0]:
            best_PINN_model = PINN_val_loss_evolution[-1], copy.deepcopy(PINN_model).cpu()
        if best_regular_model is None or regular_val_loss_evolution[-1] < best_regular_model[0]:
            best_regular_model = regular_val_loss_evolution[-1], copy.deepcopy(regular_model).cpu()

        print()
        print("Train loss: ", PINN_train_loss_epoch[-1])
        print("Val loss: ", PINN_val_loss_evolution[-1])
        print()


    plt.ioff()

    fig2, ax2 = plt.subplots()
    ax2.semilogy(regular_val_loss_evolution, label="NN loss")
    ax2.semilogy(PINN_val_loss_evolution, label="PINN loss")
    ax2.set(title="Validation loss evolution", xlabel="# epochs", ylabel="Loss")
    ax2.legend()

    if interactive:
        plt.show()

    if results_folder is not None:
        ax.clear()
        ax.set(title="Train loss evolution", xlabel="# step", ylabel="Loss")
        ax.semilogy(NN_train_loss_epoch, label="NN loss")
        ax.semilogy(PINN_train_loss_epoch, label="PINN train loss")
        ax.semilogy(PINN_physics_loss_epoch, label="PINN physics loss")
        ax.legend()
        fig.savefig(results_folder / "train_loss.png")
        fig2.savefig(results_folder / "val loss.png")

    return best_PINN_model[1], PINN_model, best_regular_model[1], regular_model