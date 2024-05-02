import math
from pathlib import Path

import numpy as np
import torch
torch.manual_seed(42)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

from src.visualization.field import save_field_animation
from src.pinns.paper.model import get_PINN_warmup_loss_fn, show_predictions, L4loss
from src.pinns.paper.train import train_batched
from models import MLPWithGeometry
import src.pinns.paper.model as m

m.IMG_SIZE = (284, 250)

RESULTS_FOLDER = Path("results/rail_sample_1ns_all_geometry")
IMG_SIZE = (284, 250)
SPATIAL_DOMAIN_SIZE = ((0, 1.5), (0, 1.7))
EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
DEVICE = "cuda:2"
LR = 0.001
RNG = np.random.default_rng(42)
COLLOCATION_DOMAIN_TIME_START = 20e-10
COLLOCATION_DOMAIN_SIZE = (180e-10, 1.7, 1.5)
EPOCHS_WARMUP = 100
EPOCHS = 300
BATCH_SIZE_WARMUP = 8192
BATCH_SIZE = 8192
N_COLLOCATION_POINTS = 8192

class PowNormalizer():
    def __init__(self, root = 3):
        self.root = root

    def fit(self, data: torch.Tensor, labels: torch.Tensor):
        """
        data: 2D tensor of shape [N samples, N features], order is x, y, t
        labels: 1D tensor of labels
        """
        self.data_scale = data.abs().max(dim=0).values
        rooted_labels = self._scale_power(labels, 1 / self.root)
        self.label_scale = rooted_labels.abs().max()

        print("Data scale:", self.data_scale)
        print("Label scale:", self.label_scale)
    
    def transform(self, data: torch.Tensor | None, labels: torch.Tensor | None):
        """
        data: 2D tensor of shape [N features, n_samples], order is x, y, t
        labels: 1D tensor of labels
        """
        if data is not None:
            data = (data.T / self.data_scale).T
        if labels is not None: 
            labels = self._scale_power(labels, 1 /self.root)
            labels = labels / self.label_scale
        
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
            labels = self._scale_power(labels, 1 / self.root)
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
            labels = labels * self.label_scale
            labels = labels.pow(self.root)
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
            labels.pow(self.root)
        return x, y, t, labels

    def _scale_power(self, labels: torch.Tensor, power: float):
        negative = labels < 0
        labels = labels.abs()
        labels = torch.pow(labels, power)
        labels[negative] *= -1
        return labels
    
    def to(self, device: str):
        self.data_scale = self.data_scale.to(device)
        self.label_scale = self.label_scale.to(device)

class PaperDataset(Dataset):
    def __init__(self, snapshots: np.ndarray, t_offsets: list[float], scaler: PowNormalizer = None):
        # input array has shape [t, y, x] -> gets flattened in x -> y -> t
        self.snapshots_shape = snapshots.shape
        self.snapshots = snapshots.flatten()
        self.t_offsets = t_offsets
        data = []
        labels = []
        for index in range(len(self.snapshots)):
            x = index % self.snapshots_shape[2]
            x *= 0.006
            y = (index // self.snapshots_shape[2]) % self.snapshots_shape[1]
            y *= 0.006
            t = self.t_offsets[index // (self.snapshots_shape[1] * self.snapshots_shape[2])] * 1e-10
            u = self.snapshots[index]
            data.append((x, y, t))
            labels.append(u)
        data = np.array(data)
        labels = np.array(labels)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        # labels = self.scale_power(labels, 1/3)
        if scaler is None:
            self.scaler = PowNormalizer()
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
    
    def print_info(self):
        print("Data shape:", self.data.shape)
        print("Data min:", self.data.min(dim=0).values)
        print("Data max:", self.data.max(dim=0).values)
        print("Label shape:", self.labels.shape)
        print("Label min:", self.labels.min(dim=0).values)
        print("Label max:", self.labels.max(dim=0).values)
        print()

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
    t1 = t*1e9
    return 1 / (201.63 * torch.e**(-0.2396*(t1 + -7.759)))**(1/3)

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
        tc = tc + COLLOCATION_DOMAIN_TIME_START
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



        # print("Term1:", term1)
        # print("Term2:", term2)
        # print("Term3:", term3)

        # collocation_loss = dftt - (1/(epsilon*mu)) * (dfxx + dfyy) + dft * sigma / epsilon
        collocation_loss = term1 + term2 + term3
        # plt.ioff()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # mappable = ax.scatter(xc.cpu().detach().numpy(), yc.cpu().detach().numpy(), tc.cpu().detach().numpy(), s = 10, c = (term1 - term2).cpu().detach().numpy())
        # plt.colorbar(mappable)
        # plt.show()
        collocation_loss = l(2e-20 * collocation_loss, torch.zeros_like(collocation_loss))


        physics_loss = collocation_loss

        return train_loss, physics_loss

    return loss_fn

def debug_geometry(geometry):
    xs = torch.linspace(0, 1.5, steps=750)
    ys = torch.linspace(0, 1.7, steps=850)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), geometry[0].numpy())
    plt.show()

    xs = torch.linspace(0, 1.5, steps=100)
    ys = torch.linspace(0, 1.7, steps=100)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    EM_values = get_EM_values(x, y, geometry)
    z = EM_values[0]
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
    plt.show()

def get_f(model: MLPWithGeometry, scaler:PowNormalizer, geometry: torch.Tensor):
    g = geometry[:3]

    def f(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x, y, t, _ = scaler.transform_(x, y, t, None)

        mlp_inputs = torch.stack([x, y, t], dim=-1) 

        u : torch.Tensor = model.forward_common_geometry(mlp_inputs, g)
        u.squeeze()

        _, u = scaler.inverse_transform(None, u)
        return u.squeeze()

    return f

def rail_sample():

    (RESULTS_FOLDER / "warmup").mkdir(exist_ok=True, parents=True)

    snapshots = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00000/snapshots.npz")["00000_E"]
    geometry = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00000/scan_00000_geometry.npy")

    # snapshots = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00000/snapshots.npz")["00000_E"][1:]
    # geometry = np.load("dataset_ascan_snapshots_0.1ns/output/scan_00000/scan_00000_geometry.npy")

    geometry = block_reduce(geometry, block_size=(1, 3, 3), func=np.mean)
    geometry = torch.from_numpy(geometry).to(DEVICE)

    # plt.imshow(geometry[1].cpu())
    # plt.show()

    # geometry = geometry.cpu()
    # EM_values = get_EM_values(torch.Tensor([0., 0., 1.45, 1.45]), torch.tensor([0., 1.6, 0., 1.6]), geometry)
    # print(EM_values)
    # debug_geometry(geometry)


    # define models and optimizers
    PINN_model = MLPWithGeometry()
    # PINN_model.load_state_dict(torch.load("results/rail_sample_02ns_all_pow/NN_model_best.ckp"))
    PINN_model = PINN_model.to(DEVICE)

    regular_model = MLPWithGeometry()
    # regular_model.load_state_dict(torch.load("results/rail_sample_05ns/NN_model_best_warmup.ckp"))
    regular_model = regular_model.to(DEVICE)

    PINN_optimizer = torch.optim.Adam(PINN_model.parameters(), lr = LR)
    regular_optimizer = torch.optim.Adam(regular_model.parameters(), lr = LR)


    # Create the dataset
    train_indexes = list(range(20, len(snapshots), 10))
    print(train_indexes)
    train_dataset = PaperDataset(snapshots[train_indexes], t_offsets=train_indexes)
    print("Train dataset points:")
    train_dataset.print_info()
    save_field_animation(train_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=200)
    # frame_15ns = train_dataset.get_frame(1)
    #show_field(frame_15ns)
    scaler = train_dataset.scaler

    val_indexes = list(range(21, len(snapshots), 10))
    val_dataset = PaperDataset(snapshots[val_indexes], t_offsets=val_indexes, scaler=scaler)
    print("Val dataset points:")
    val_dataset.print_info()
    save_field_animation(val_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=200)

    test_indexes = list(range(25, len(snapshots), 10))
    test_dataset = PaperDataset(snapshots[test_indexes], t_offsets=test_indexes, scaler=scaler)
    print("Test dataset points:")
    test_dataset.print_info()
    save_field_animation(test_dataset.snapshots.reshape((-1, *IMG_SIZE)), None, interval=200)

    # get the derivative functions
    f_PINN = get_f(PINN_model, scaler, geometry)
    PINN_loss_fn_L4 = get_PINN_warmup_loss_fn(L4loss)
    PINN_loss_fn_L2 = get_PINN_uniform_loss_fn(nn.MSELoss())
    regular_loss_fn = nn.MSELoss()

    # get f for regular model
    f_regular = get_f(regular_model, scaler, geometry)
    #plot_data_histogram(train_dataset.data, train_dataset.labels)


    # print("collocation_points:")
    # print("min:", collocation_points.min(dim=1))
    # print("max:", collocation_points.max(dim=1))

    # generate dataset
    train_loader = DataLoader(train_dataset, BATCH_SIZE_WARMUP, shuffle=True, num_workers=64, persistent_workers=True)

    val_samples = torch.cat([val_dataset.data, val_dataset.labels[:, None]], dim=1).T.to(DEVICE)
    test_samples = torch.cat([test_dataset.data, test_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    if EPOCHS_WARMUP > 0:
        best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train_batched(PINN_model,
                                                    f_PINN,
                                                    PINN_optimizer,
                                                    PINN_loss_fn_L4,
                                                    regular_model,
                                                    f_regular,
                                                    regular_optimizer,
                                                    L4loss,
                                                    train_loader,
                                                    val_samples,
                                                    None,
                                                    None,
                                                    geometry,
                                                    EPOCHS_WARMUP,
                                                    DEVICE,
                                                    results_folder=RESULTS_FOLDER / "warmup",
                                                    interactive=False)

        best_PINN_model = best_PINN_model.to(DEVICE)
        best_regular_model = best_regular_model.to(DEVICE)
        regular_optimizer = torch.optim.Adam(best_regular_model.parameters(), lr = LR)

        f_PINN = get_f(best_PINN_model, scaler, geometry)
        f_regular = get_f(best_regular_model, scaler, geometry)
        PINN_optimizer = torch.optim.Adam(best_PINN_model.parameters(), lr = LR)
        torch.save(best_PINN_model.state_dict(), RESULTS_FOLDER / "PINN_model_best_warmup.ckp")
        torch.save(best_regular_model.state_dict(), RESULTS_FOLDER / "NN_model_best_warmup.ckp")
    else:
        best_PINN_model = PINN_model
        best_regular_model = regular_model

    for i, val_snapshot in enumerate(torch.split(val_samples, math.prod(IMG_SIZE), dim=1)):
        show_predictions(f_PINN, f_regular, val_snapshot, RESULTS_FOLDER / f"warmup/val_predictions_{i}.png")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=64, persistent_workers=True)

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
                                                interactive=False)

    torch.save(best_PINN_model.state_dict(), RESULTS_FOLDER / "PINN_model_best.ckp")
    torch.save(best_regular_model.state_dict(), RESULTS_FOLDER / "NN_model_best.ckp")

    best_PINN_model = best_PINN_model.to(DEVICE)
    best_regular_model = best_regular_model.to(DEVICE)

    for i, val_snapshot in enumerate(torch.split(val_samples, math.prod(IMG_SIZE), dim=1)):
        show_predictions(f_PINN, f_regular, val_snapshot, RESULTS_FOLDER / f"val_predictions_{i}.png")
    for i, test_snapshot in enumerate(torch.split(test_samples, math.prod(IMG_SIZE), dim=1)):
        show_predictions(f_PINN, f_regular, test_snapshot, RESULTS_FOLDER / f"test_predictions_{i}.png")

if __name__ == "__main__":
    rail_sample()

