import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur

from src.pinns.models.base import  MLP

class UpConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, activation: nn.Module):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.activation = activation()

    def forward(self, inputs):
        x = self.up(inputs)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)

        return x

class UpCNN(nn.Module):   
    def __init__(self,
                 channels_size: list[int],
                 activations: nn.Module):
        super().__init__()

        conv_blocks = []
        for i in range(len(channels_size) - 1):
            conv_blocks.append(UpConvBlock(channels_size[i], channels_size[i+1], activations))

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.regressor = nn.Conv2d(channels_size[-1], 1, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_blocks(x)
        x = self.regressor(x)
        return x

class Time2Image(nn.Module):
    def __init__(self, fc_layers: list[int] = [1, 64, 256, 1152], bottleneck_size = (36, 32), cnn_layers: list[int] = [1, 64, 64, 16], activations: nn.Module = nn.ReLU):
        super().__init__()

        self.bottleneck_size = bottleneck_size
        self.mlp = MLP(1, fc_layers[1:-1], fc_layers[-1], activations)
        self.up_cnn = UpCNN(cnn_layers, activations)
        self.blur = GaussianBlur(5, 2)
    
    def forward(self, x: torch.Tensor):
        x = self.mlp(x)

        x = x.reshape(-1, 1, *self.bottleneck_size)

        x = self.up_cnn(x)

        # x = self.blur(x)

        return x
    
if __name__ == "__main__":
    print("Time2Image:")
    inputs = torch.tensor([[0.], [1.], [2.]])
    print("inputs:", inputs.shape)

    upnet = Time2Image()
    out = upnet(inputs)
    print(out.shape)
