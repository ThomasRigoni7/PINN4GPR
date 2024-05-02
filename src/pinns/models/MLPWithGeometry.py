"""
This module contains the implementation of a combined MLP and CNN architecture, 
which takes as input the x, y, t coordinates and the full input geometry and outputs
the electric field at that point.

I was not able to make the model predict reasonable fields, most probably due to the less than perfect feature fusion.
"""


import torch
import torch.nn as nn
from torchvision.transforms import Pad

from src.pinns.models.base import  MLP

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module,
                 use_batch_norm: bool,
                 kernel_size: int = 3,
                 padding: tuple[int,int] | int = 1):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation())
        layers.append(nn.MaxPool2d(2))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

class CNN(nn.Module):   
    def __init__(self,
                 in_shape: tuple[int],
                 channels_size: list[int],
                 bottleneck_nodes: int,
                 activations: nn.Module):
        super().__init__()

        conv_blocks = [ConvBlock(in_shape[0], channels_size[0], activations, True)]
        for i in range(len(channels_size) - 1):
            conv_blocks.append(ConvBlock(channels_size[i], channels_size[i+1], activations, True))

        self.conv_blocks = nn.Sequential(*conv_blocks)

        final_height = int(in_shape[1] / (2**((len(channels_size)))))
        final_width = int(in_shape[2] / (2**((len(channels_size)))))
        linear_nodes = channels_size[-1] * final_height * final_width
        self.linear = nn.Linear(linear_nodes, bottleneck_nodes)

    def forward(self, x: torch.Tensor):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)
        return self.linear(x)

class MLPWithGeometry(nn.Module):
    def __init__(self, 
                 in_shape: tuple[int, int, int] = (3, 284, 250),
                 cnn_channels: list[int] = [8, 16, 32, 16, 8],
                 cnn_out_nodes: int = 64,
                 bottleneck_nodes: int = 128,
                 img_padding: tuple[int, int] = (3, 2), 
                 mlp_inputs: int = 3,
                 mlp_layer_sizes: list[int] = [128]*4,
                 mlp_outputs: int = 1,
                 activations: nn.Module = nn.SiLU) -> None:
        super().__init__()

        self.padding = Pad(img_padding, padding_mode="reflect")

        padded_input_shape = (in_shape[0], in_shape[1] + 2*img_padding[1], in_shape[2] + 2*img_padding[0])
        self.cnn = CNN(padded_input_shape, cnn_channels, cnn_out_nodes, activations)

        self.input_linear = nn.Sequential(nn.Linear(mlp_inputs, bottleneck_nodes), activations())
        self.mlp = MLP(cnn_out_nodes + bottleneck_nodes, mlp_layer_sizes, mlp_outputs, activations)

    def forward(self, mlp_inputs: torch.Tensor, geometry: torch.Tensor):
        """
        regular forward with batched mlp inputs and geometries
        """
        padded_geometry = self.padding(geometry)
        cnn_embeddings = self.cnn(padded_geometry)
        linear_embeddings = self.input_linear(mlp_inputs)
        x = torch.cat([linear_embeddings, cnn_embeddings], dim=-1)
        return self.mlp(x)
    
    def forward_cnn_embeddings(self, mlp_inputs: torch.Tensor, geometry_embeddings: torch.Tensor):
        """
        forward with batched mlp inputs and geometry embeddings, skips the CNN.
        """
        linear_embeddings = self.input_linear(mlp_inputs)
        x = torch.cat([linear_embeddings, geometry_embeddings], dim=-1)
        return self.mlp(x)
    
    def forward_common_geometry(self, mlp_inputs: torch.Tensor, geometry: torch.Tensor):
        """
        forward with batched mlp inputs and single geometry: 

        extracts the embeddings from the geometry and broadcasts them as input to the mlp.
        """

        assert geometry.ndim == 3, f"Common geometry must have 3 dimensions, given shape: {geometry.shape}"

        padded_geometry = self.padding(geometry)
        cnn_embeddings = self.cnn(padded_geometry.unsqueeze(0))
        linear_embeddings = self.input_linear(mlp_inputs)

        # broadcasting
        batch_size = mlp_inputs.shape[0]
        cnn_embeddings = torch.broadcast_to(cnn_embeddings.squeeze(), (batch_size, cnn_embeddings.shape[-1]))

        x = torch.cat([linear_embeddings, cnn_embeddings], dim=-1)
        return self.mlp(x)
    
if __name__ == "__main__":
    import numpy as np
    from skimage.measure import block_reduce

    snapshots = np.load("dataset/gprmax_output_files/scan_00000/snapshots.npz")["00000_E"]
    geometry = np.load("dataset/gprmax_output_files/scan_00000/scan_00000_geometry.npy")

    geometry = block_reduce(geometry, block_size=(1, 3, 3), func=np.mean)

    snapshots = torch.from_numpy(snapshots)
    geometry = torch.from_numpy(geometry)
    print("snapshots:", snapshots.shape)
    print("Geometry:", geometry.shape)

    print("PINN4GPR:")
    inputs = torch.tensor([[0., 0., 0.], [0.2, 0.2, 0.2]])
    geometries = torch.stack([geometry[:3]]*2)
    print("inputs shape:", inputs.shape)
    print("input geom shape:", geometries.shape)

    net = MLPWithGeometry()
    out = net(inputs, geometries)

    print(out)