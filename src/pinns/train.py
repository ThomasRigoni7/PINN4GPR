"""
This module contains training scripts
"""

from tqdm import tqdm
import copy
from typing import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


LOGGING_FREQUENCY = 1

def evaluate_batch(forward_fn: Callable,
                   loader: DataLoader,
                   device: str,
                   loss_fn: Callable = nn.MSELoss(reduction="sum")):
    losses = []
    with torch.no_grad:
        for batch in loader:
            
            inputs = batch[:-1]
            labels = batch[-1]

            device_inputs = []
            for i in inputs:
                device_inputs.append(i.to(device))
            
            labels = labels.to(device)
            preds = forward_fn(inputs)
            loss = loss_fn(preds, labels) / labels.shape[0]
            losses.append(float(loss))

    return torch.tensor(losses).mean()
    

def train_batched(model: nn.Module,
          forward_fn: Callable,
          optimizer: torch.optim.Optimizer,
          loss_fn: Callable,
          train_loader: DataLoader,
          val_loader: torch.Tensor,
          epochs: int,
          device: str,
          use_scheduler:bool = False,
          results_folder: str | Path = None,
          interactive: bool = True):
    
    if use_scheduler:
        scheduler = StepLR(optimizer, 1000, 0.9)


    fig, ax = plt.subplots()
    if interactive:
        plt.ion()


    train_loss_step = []
    physics_loss_step = []

    train_loss_epoch = []
    physics_loss_epoch = []

    best_model = None

    val_loss_evolution = []
    with torch.enable_grad():
        for e in tqdm(range(epochs), position=0, desc="Epoch"):
            for batch in tqdm(train_loader, position=1, desc="Step", leave=None):
                optimizer.zero_grad()

                inputs = batch[:-1]
                labels = batch[-1]

                device_inputs = []
                for i in inputs:
                    device_inputs.append(i.to(device))
                
                labels = labels.to(device)

                loss_output = loss_fn(forward_fn, inputs, labels)

                if isinstance(loss_output, list) and len(loss_output) == 2:
                    train_loss, physics_loss = loss_output
                    loss = train_loss + physics_loss
                    train_loss_step.append(float(train_loss))
                    physics_loss_step.append(float(physics_loss))
                else:
                    loss = loss_output

                loss.backward()
                optimizer.step()

                if use_scheduler:
                    scheduler.step()

            # compute validation loss
            val_loss_evolution.append(evaluate_batch(forward_fn, val_loader, device))

            # calculate the per epoch losses
            train_loss_epoch.append(float(torch.tensor([train_loss_step]).mean()))
            physics_loss_epoch.append(float(torch.tensor([physics_loss_step]).mean()))

            train_loss_step = []
            physics_loss_step = []

            # draw
            ax.clear()
            ax.set(title="Train loss evolution", xlabel="# epoch", ylabel="Loss")
            ax.semilogy(train_loss_epoch, label="train loss")
            ax.semilogy(physics_loss_epoch, label="physics loss")
            ax.semilogy(val_loss_evolution, label="val loss")
            ax.set_ylim(1, None)
            ax.legend()
            if interactive:
                plt.show()
                plt.pause(0.01)

            # check best model
            if best_model is None or val_loss_evolution[-1] < best_model[0]:
                best_model = val_loss_evolution[-1], copy.deepcopy(model).cpu()


    plt.ioff()

    fig2, ax2 = plt.subplots()
    ax2.semilogy(val_loss_evolution, label="val loss")
    ax2.set(title="Validation loss evolution", xlabel="# epochs", ylabel="Loss")
    ax2.legend()

    if interactive:
        plt.show()

    if results_folder is not None:
        ax.clear()
        ax.set(title="Train loss evolution", xlabel="# step", ylabel="Loss")
        ax.semilogy(train_loss_epoch, label="train loss")
        ax.semilogy(physics_loss_epoch, label="physics loss")
        ax.legend()
        fig.savefig(results_folder / "train_loss.png")
        fig2.savefig(results_folder / "val_loss.png")

    return best_model[1], model