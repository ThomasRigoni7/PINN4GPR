from pathlib import Path

import numpy as np
import torch
torch.manual_seed(42)
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from src.visualization.field import save_field_animation
from src.pinns.paper.dataset import MyNormalizer, PaperDataset
from src.pinns.paper.model import MLP, IMG_SIZE, get_f, get_PINN_warmup_loss_fn, show_predictions, get_EM_values
from src.pinns.paper.train import train_batched
import src.pinns.paper.model as m

m.IMG_SIZE = (284, 250)

RESULTS_FOLDER = Path("results/two_layers_batched")
IMG_SIZE = (284, 250)
SPATIAL_DOMAIN_SIZE = ((0, 1.5), (0, 1.7))
EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
DEVICE = "cuda:2"
LR = 0.001
RNG = np.random.default_rng(42)
COLLOCATION_DOMAIN_SIZE = (7e-9, 1.7, 1.5)
EPOCHS_WARMUP = 30
EPOCHS = 10
BATCH_SIZE = 512
N_COLLOCATION_POINTS = 4096


def get_EM_values(x: torch.Tensor, y: torch.Tensor, geometry: torch.Tensor):
    """
    Returns the EM values of the point, given its coordinates and the geometry
    """

    percent_x = (x - SPATIAL_DOMAIN_SIZE[0][0]) / (SPATIAL_DOMAIN_SIZE[0][1] - SPATIAL_DOMAIN_SIZE[0][0])
    index_x = percent_x * (geometry.shape[2])
    percent_y = (y - SPATIAL_DOMAIN_SIZE[1][0]) / (SPATIAL_DOMAIN_SIZE[1][1] - SPATIAL_DOMAIN_SIZE[1][0])
    index_y = percent_y * (geometry.shape[1])

    index_x = torch.clamp(index_x, 0, geometry.shape[2] - 1)
    index_y = torch.clamp(index_y, 0, geometry.shape[1] - 1)

    return geometry[:, index_y.int(), index_x.int()]

def get_time_weights(t):
    return 1 / (252.86 * np.e**(-0.2388*(t + -4.8314)))

def get_PINN_uniform_loss_fn(training_points_loss_fn):
    def loss_fn(f, x, y, t, u, _b, _c, geometry: torch.Tensor):
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
        train_error = train_preds - u
        # time_weights = get_time_weights(t)
        time_weights = 1
        train_loss = training_points_loss_fn(train_error * time_weights, torch.zeros_like(train_error))

        # collocation points
        l = nn.MSELoss()

        # collocation points
        collocation_points = RNG.uniform(size=(3, N_COLLOCATION_POINTS)) * np.array(COLLOCATION_DOMAIN_SIZE).reshape(3, -1)
        tc, yc, xc = collocation_points
        # tc = tc + train_indexes[0] * 1e-9
        collocation_points = np.stack([xc, yc, tc])
        collocation_points = torch.tensor(collocation_points, dtype=torch.float32)
        collocation_points = collocation_points.to(DEVICE)
        xc, yc, tc = collocation_points

        EM_values = get_EM_values(xc, yc, geometry)

        epsilon, sigma, mu, _ = EM_values

        # print("eps:", epsilon.min(), epsilon.max())
        # print("sigma:", sigma.min(), sigma.max())
        # print("mu:", mu.min(), mu.max())

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



        # print("Term1:", term1.abs().max())
        # print("Term2:", term2.abs().max())
        # print("Term3:", term3.abs().max())

        # collocation_loss = dftt - (1/(epsilon*mu)) * (dfxx + dfyy) + dft * sigma / epsilon
        collocation_loss = term1 + term2 + term3
        # plt.ioff()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # mappable = ax.scatter(xc.cpu().detach().numpy(), yc.cpu().detach().numpy(), tc.cpu().detach().numpy(), s = 10, c = (term1 - term2).cpu().detach().numpy())
        # plt.colorbar(mappable)
        # plt.show()
        # print(collocation_loss.mean())
        collocation_loss = l(3e-19 * collocation_loss, torch.zeros_like(collocation_loss))


        physics_loss = collocation_loss

        return train_loss, physics_loss

    return loss_fn

def two_layer():
    """
    Trains a two-layer model PINN and NN and compares them.
    """

    snapshots = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00000/snapshots.npz")["00000_E"]

    geometry = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00000/scan_00000_geometry.npy")
    geometry = torch.from_numpy(geometry).to(DEVICE)

    # print(geometry.shape)
    # plt.imshow(geometry[0].cpu())
    # plt.show()

    # print(snapshots.shape)
    # plt.imshow(snapshots[0])
    # plt.show()


    # define models and optimizers
    PINN_model = MLP(3, [256]*5, 1, nn.SiLU)
    # PINN_model.load_state_dict(torch.load("results/two_layers_batched/PINN_model_best_warmup.ckp"))
    PINN_model = PINN_model.to(DEVICE)

    regular_model = MLP(3, [256]*5, 1, nn.ReLU)
    regular_model = regular_model.to(DEVICE)

    PINN_optimizer = torch.optim.Adam(PINN_model.parameters(), lr = LR)
    regular_optimizer = torch.optim.Adam(regular_model.parameters(), lr = LR)


    # Create the dataset
    train_indexes = [0, 2, 4, 6]
    train_dataset = PaperDataset(snapshots[train_indexes], t_offsets=train_indexes)
    print("Train dataset points:")
    train_dataset.print_info()
    # save_field_animation(train_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=50)
    # frame_15ns = train_dataset.get_frame(1)
    #show_field(frame_15ns)
    scaler = train_dataset.scaler
    val_indexes = [2]
    val_dataset = PaperDataset(snapshots[val_indexes], t_offsets=val_indexes, scaler=scaler)
    print("Validation dataset points:")
    val_dataset.print_info()
    # save_field_animation(val_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=50)

    test_indexes = [5]
    test_dataset = PaperDataset(snapshots[test_indexes], t_offsets=test_indexes, scaler=scaler)
    print("Test dataset points:")
    test_dataset.print_info()
    # save_field_animation(test_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=50)

    # get the derivative functions
    f_PINN = get_f(PINN_model, scaler)
    PINN_loss_fn_L4 = get_PINN_warmup_loss_fn(nn.MSELoss())
    PINN_loss_fn_L2 = get_PINN_uniform_loss_fn(nn.MSELoss())
    regular_loss_fn = nn.MSELoss()

    # get f for regular model
    f_regular = get_f(regular_model, scaler)
    #plot_data_histogram(train_dataset.data, train_dataset.labels)

    # generate dataset
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=64)
    print("Len train dataset:", len(train_dataset))
    print("Len train loader:", len(train_loader))

    print("building validation dataset...")
    val_samples = torch.cat([val_dataset.data, val_dataset.labels[:, None]], dim=1).T.to(DEVICE)


    print("building test dataset...")
    test_samples = torch.cat([test_dataset.data, test_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    if EPOCHS_WARMUP > 0:
        best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train_batched(PINN_model,
                                                    f_PINN,
                                                    PINN_optimizer,
                                                    PINN_loss_fn_L4,
                                                    regular_model,
                                                    f_regular,
                                                    regular_optimizer,
                                                    nn.MSELoss(),
                                                    train_loader,
                                                    val_samples,
                                                    None,
                                                    None,
                                                    geometry,
                                                    EPOCHS_WARMUP,
                                                    DEVICE,
                                                    results_folder=RESULTS_FOLDER / "warmup",
                                                    interactive=True)

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

    best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train_batched(best_PINN_model,
                                                f_PINN,
                                                PINN_optimizer,
                                                PINN_loss_fn_L2,
                                                best_regular_model,
                                                f_regular,
                                                regular_optimizer,
                                                regular_loss_fn,
                                                train_loader,
                                                val_samples,
                                                None,
                                                None,
                                                geometry,
                                                EPOCHS,
                                                DEVICE,
                                                use_scheduler=True,
                                                results_folder=RESULTS_FOLDER,
                                                interactive=True)

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