import math

import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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
        data: 2D tensor of shape [N features, n_samples], order is x, y, t
        labels: 1D tensor of labels
        """
        if data is not None:
            data = (data.T / self.data_scale).T
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
    
    def to(self, device: str):
        self.data_scale = self.data_scale.to(device)
        self.label_scale = self.label_scale.to(device)

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
    
    def print_info(self):
        print("Data shape:", self.data.shape)
        print("Data min:", self.data.min(dim=0).values)
        print("Data max:", self.data.max(dim=0).values)
        print("Label shape:", self.labels.shape)
        print("Label min:", self.labels.min(dim=0).values)
        print("Label max:", self.labels.max(dim=0).values)
        print()

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