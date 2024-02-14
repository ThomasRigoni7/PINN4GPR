from pathlib import Path

import numpy as np
import torch
torch.manual_seed(42)
import torch.nn as nn
import cv2

import matplotlib.pyplot as plt
from src.visualization.misc import save_field_animation
from src.pinns.paper.dataset import MyNormalizer, PaperDataset
from src.pinns.paper.model import MLP, IMG_SIZE, get_f, get_PINN_warmup_loss_fn, get_PINN_loss_fn, L4loss, show_predictions, get_EM_values
from src.pinns.paper.train import train


def two_layer():
    """
    Trains a two-layer model PINN and NN and compares them.
    """

    DEVICE = "cuda:2"
    LR = 0.001
    RNG = np.random.default_rng(42)
    N_COLLOCATION_POINTS = 40000
    N_BOUNDARY_POINTS = 5000
    COLLOCATION_DOMAIN_SIZE = (35e-9, 20, 20)
    EPOCHS_WARMUP = 5000
    EPOCHS = 20000
    RESULTS_FOLDER = Path("results/two_layer")

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
    geometry = torch.from_numpy(geometry).to(DEVICE)


    # define models and optimizers
    PINN_model = MLP(3, [256]*5, 1, nn.SiLU)
    # PINN_model.load_state_dict(torch.load("checkpoints/PINN_model_best_warmup_2layers_15_21.ckp"))
    PINN_model = PINN_model.to(DEVICE)

    regular_model = MLP(3, [256]*5, 1, nn.ReLU)
    regular_model = regular_model.to(DEVICE)

    PINN_optimizer = torch.optim.Adam(PINN_model.parameters(), lr = LR)
    regular_optimizer = torch.optim.Adam(regular_model.parameters(), lr = LR)


    # Create the dataset
    train_indexes = [15, 19, 23, 27, 31, 35]
    train_dataset = PaperDataset(snapshots[train_indexes], t_offsets=train_indexes)
    print("Train dataset points:")
    train_dataset.print_info()
    save_field_animation(train_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=50)
    # frame_15ns = train_dataset.get_frame(1)
    #show_field(frame_15ns)
    scaler = train_dataset.scaler
    val_indexes = [25]
    val_dataset = PaperDataset(snapshots[val_indexes], t_offsets=val_indexes, scaler=scaler)
    print("Validation dataset points:")
    val_dataset.print_info()

    test_indexes = [45]
    test_dataset = PaperDataset(snapshots[test_indexes], t_offsets=test_indexes, scaler=scaler)
    print("Test dataset points:")
    test_dataset.print_info()
    # save_field_animation(test_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=50)
    # frame_60ns = val_dataset.get_frame(0)
    # show_field(frame_60ns)


    # collocation points
    collocation_points = RNG.uniform(size=(3, N_COLLOCATION_POINTS)) * np.array(COLLOCATION_DOMAIN_SIZE).reshape(3, -1)
    tc, yc, xc = collocation_points
    tc = tc + train_indexes[0] * 1e-9
    collocation_points = np.stack([xc, yc, tc])
    collocation_points = torch.tensor(collocation_points, dtype=torch.float32)
    # collocation_points, _ = scaler.transform(collocation_points.T, None)
    collocation_points = collocation_points.to(DEVICE)

    print("collocation points:", collocation_points.shape)
    print("min:", collocation_points.min(axis=1).values)
    print("max:", collocation_points.max(axis=1).values)

    # boundary points
    # TODO: works only if x and y size are the same
    boundary_points = RNG.uniform(size=(2, N_BOUNDARY_POINTS)) * np.array((COLLOCATION_DOMAIN_SIZE[1], COLLOCATION_DOMAIN_SIZE[0])).reshape(2, -1)
    print("bound shape:", boundary_points.shape)
    print("min:", boundary_points.min(axis=1))
    print("max:", boundary_points.max(axis=1))

    boundary_points = np.stack([boundary_points[0, :N_BOUNDARY_POINTS], np.ones(N_BOUNDARY_POINTS) * 15.0, boundary_points[1, :N_BOUNDARY_POINTS] + 1.5e-8])

    boundary_points = torch.from_numpy(boundary_points)
    boundary_points = boundary_points.to(DEVICE, torch.float32)

    print("boundary points:", boundary_points.shape)
    print("min:", boundary_points.min(axis=1).values)
    print("max:", boundary_points.max(axis=1).values)

    # get the derivative functions
    f_PINN = get_f(PINN_model, scaler)
    PINN_loss_fn_L4 = get_PINN_warmup_loss_fn(L4loss)
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

    if EPOCHS_WARMUP > 0:
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
                                                    EPOCHS_WARMUP,
                                                    results_folder=RESULTS_FOLDER / "warmup",
                                                    interactive=False)

        best_PINN_model = best_PINN_model.to(DEVICE)
        best_regular_model = best_regular_model.to(DEVICE)
        regular_optimizer = torch.optim.Adam(best_regular_model.parameters(), lr = LR)

        f_PINN = get_f(best_PINN_model, scaler)
        f_regular = get_f(best_regular_model, scaler)
        PINN_optimizer = torch.optim.Adam(best_PINN_model.parameters(), lr = LR)
        torch.save(best_PINN_model.state_dict(), RESULTS_FOLDER / "PINN_model_best_warmup.ckp")
    else:
        best_PINN_model = PINN_model
        best_regular_model = regular_model

    show_predictions(f_PINN, f_regular, val_samples, RESULTS_FOLDER / "warmup/val_predictions.png")

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
                                                EPOCHS,
                                                use_scheduler=True,
                                                results_folder=RESULTS_FOLDER,
                                                interactive=False)

    torch.save(best_PINN_model.state_dict(), RESULTS_FOLDER / "PINN_model_best.ckp")
    torch.save(best_regular_model.state_dict(), RESULTS_FOLDER / "NN_model_best.ckp")

    best_PINN_model = best_PINN_model.to(DEVICE)
    best_regular_model = best_regular_model.to(DEVICE)

    show_predictions(f_PINN, f_regular, val_samples, RESULTS_FOLDER / "val_predictions.png")
    show_predictions(f_PINN, f_regular, test_samples, RESULTS_FOLDER / "test_predictions.png")

if __name__ == "__main__":
    two_layer()


# 5x64
# PINN train loss 0.04249459132552147
# PINN val loss 0.1067247986793518
# PINN physics loss 0.4107848107814789
# NN train loss 1.9267771244049072
# NN val loss 2.8269050121307373
    
# 5x256 still no reflection
# PINN train loss 0.0013126502744853497
# PINN val loss 0.00221841549500823
# PINN physics loss 0.006338852923363447
# NN train loss 0.07565296441316605
# NN val loss 0.08619776368141174