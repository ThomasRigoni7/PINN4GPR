import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur

from src.pinns.models.base import  MLP

class UpConvBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, activation: nn.Module):
        super().__init__()

        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.activation = activation()

    def forward(self, inputs):
        x = self.up(inputs)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)

        return x

class UpCNN1D(nn.Module):   
    def __init__(self,
                 channels_size: list[int],
                 activations: nn.Module):
        super().__init__()

        conv_blocks = []
        for i in range(len(channels_size) - 1):
            conv_blocks.append(UpConvBlock1D(channels_size[i], channels_size[i+1], activations))

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.regressor = nn.Conv1d(channels_size[-1], 1, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_blocks(x)
        x = self.regressor(x)
        return x

def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
        ksize_half = (kernel_size - 1) * 0.5

        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d

class Time2Sequence(nn.Module):
    def __init__(self, fc_layers: list[int] = [1, 64, 256], bottleneck_size = (256,), cnn_layers: list[int] = [1, 64, 64, 16], activations: nn.Module = nn.ReLU):
        super().__init__()

        self.bottleneck_size = bottleneck_size
        self.mlp = MLP(1, fc_layers[1:-1], fc_layers[-1], activations)
        self.up_cnn = UpCNN1D(cnn_layers, activations)
        self.gaussian_kernel = _get_gaussian_kernel1d(7, 3.0).view(1, 1, 7)

    def blur(self, seq: torch.Tensor):
        seq = torch.nn.functional.conv1d(seq, self.gaussian_kernel, padding=3)
        return seq

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = x.reshape(-1, 1, *self.bottleneck_size)
        x = self.up_cnn(x)

        x = self.blur(x).squeeze()

        return x
    
if __name__ == "__main__":
    print("Time2Image:")
    inputs = torch.tensor([[0.], [1.], [2.]])
    print("inputs:", inputs.shape)

    upnet = Time2Sequence()
    out = upnet(inputs)
    print(out.shape)
