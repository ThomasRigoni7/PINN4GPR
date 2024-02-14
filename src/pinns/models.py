import torch
import torch.nn as nn
from torchvision.transforms import Pad

class MLP(nn.Module):
    def __init__(self, num_inputs: int, hidden_layer_sizes: list[int], num_outputs: int, activation: nn.Module):
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

    def forward(self, x):
        if x.ndim == 1:
            x = x[None, :]
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x

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

class PINN4GPR(nn.Module):
    def __init__(self, 
                 in_shape: tuple[int, int, int] = (3, 284, 250),
                 cnn_channels: list[int] = [64]*4 + [16],
                 bottleneck_nodes: int = 256,
                 img_padding: tuple[int, int] = (3, 2), 
                 mlp_inputs: int = 3,
                 mlp_layer_sizes: list[int] = [256]*5,
                 mlp_outputs: int = 1,
                 activations: nn.Module = nn.SiLU) -> None:
        super().__init__()

        self.padding = Pad(img_padding, padding_mode="reflect")

        padded_input_shape = (in_shape[0], in_shape[1] + 2*img_padding[1], in_shape[2] + 2*img_padding[0])
        self.cnn = CNN(padded_input_shape, cnn_channels, bottleneck_nodes, activations)

        self.mlp = MLP(mlp_inputs + bottleneck_nodes, mlp_layer_sizes, mlp_outputs, activations)

    def forward(self, mlp_inputs: torch.Tensor, geometry: torch.Tensor):
        padded_geometry = self.padding(geometry)
        bottleneck_nodes = self.cnn(padded_geometry)
        x = torch.cat([mlp_inputs, bottleneck_nodes], dim=-1)
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

    inputs = torch.tensor([[0., 0., 0.], [0.2, 0.2, 0.2]])
    geometires = torch.stack([geometry[:3]]*2)

    net = PINN4GPR()
    out = net(inputs, geometires)

    print(out)