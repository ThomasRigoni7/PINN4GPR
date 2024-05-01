"""
This module contains a work in progress PyTorch implementation of the black box CNN used for predicting B-scans from sample geometries.

The implementation is not equivalent to the Keras one, it does not work properly.
"""

import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = activation()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = activation()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.upsampling(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x

class ConnectBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=(1, 3))
        self.convtrans = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = activation()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.convtrans(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x
    
class MultiHeadAttention2D(nn.Module):
    def __init__(self, channels, num_heads):
        super(MultiHeadAttention2D,self).__init__()
        self.channels_in = channels
        self.activation = nn.ReLU()
        
        self.query_conv = nn.Conv2d(in_channels = channels , out_channels = channels//num_heads , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = channels , out_channels = channels//num_heads , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size= 1)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        """
            inputs :
                query :  B x C x H x W
                key   :  B x C x H x W
                value :  B x C x H x W
            returns :
                out : self attention value + input feature 
        """
        assert query.shape == key.shape == value.shape, f"""Error: query, key and value shapes are different:
        Query shape: {query.shape},
        Key shape  : {key.shape},
        Value shape: {value.shape}.
        """
        batch_size, num_channels, height, width = query.size()
        proj_query  = self.query_conv(query).view(batch_size,-1,width*height).permute(0,2,1) # B x C x (N)
        proj_key =  self.key_conv(key).view(batch_size,-1,width*height) # B x C x (W*H)
        proj_value = self.value_conv(value).view(batch_size,-1,width*height) # B x C x N

        energy =  torch.bmm(proj_query,proj_key) # transpose check
        scaled_energy = energy / torch.sqrt(torch.tensor(width*height))

        attention = self.softmax(scaled_energy) # B x N x N 
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(batch_size, num_channels, height, width)
        
        return out + value
    
class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_heads=8):
        super().__init__()
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_q = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.attention = MultiHeadAttention2D(out_channels, num_heads)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, 1)
        # TODO: add size in laayernorm
        self.layernorm = nn.LayerNorm((256, 6, 18))

    def forward(self, x1, x2):
        v1 = self.conv_v(x1)
        q2 = self.conv_q(x2)
        # self-attention: key and value are the same
        o1 = self.layernorm(q2 + v1)
        # a = self.attention(q2, v1, v1)
        # o1 = self.layernorm(a + v1)
        o2 = self.conv1x1(o1)
        x = self.layernorm(o1 + o2)
        return x

class Geom2Bscan(nn.Module):
    def __init__(self, base_channels = 16):
        super().__init__()
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16, base_channels*32]

        epsilon_doen_blocks = [DownBlock(1, channels[0]),
                       DownBlock(channels[0], channels[1]),
                       DownBlock(channels[1], channels[2]),
                       DownBlock(channels[2], channels[3]),
                       DownBlock(channels[3], channels[4])]
        self.epsilon_encoder = nn.Sequential(*epsilon_doen_blocks)

        sigma_down_blocks = [DownBlock(1, channels[0]),
                       DownBlock(channels[0], channels[1]),
                       DownBlock(channels[1], channels[2]),
                       DownBlock(channels[2], channels[3]),
                       DownBlock(channels[3], channels[4])]
        self.sigma_encoder = nn.Sequential(*sigma_down_blocks)

        self.fusion1 = CrossAttention(channels[4], channels[4])
        self.fusion2 = CrossAttention(channels[4], channels[4])
        self.connect = ConnectBlock(channels[5], channels[5])

        up_blocks = [UpBlock(channels[5], channels[4]),
                     UpBlock(channels[4], channels[3]),
                     UpBlock(channels[3], channels[2]),
                     UpBlock(channels[2], channels[1]),
                     UpBlock(channels[1], channels[0])]
        self.decoder = nn.Sequential(*up_blocks)

        self.regressor = nn.Conv2d(channels[0], 1, 1)
    
    def forward(self, epsilon_map, sigma_map):
        e1 = self.epsilon_encoder(epsilon_map)
        e2 = self.sigma_encoder(sigma_map)
        f1 = self.fusion1(e1, e2)
        f2 = self.fusion2(e2, e1)
        fused = self.connect(torch.cat([f1, f2], dim=1))
        x = self.decoder(fused)
        out = self.regressor(x)
        return out
    
if __name__ == "__main__":
    epsilon_map = torch.randn((10, 1, 200, 600))
    sigma_map = torch.randn((10, 1, 200, 600))
    model = Geom2Bscan()
    out = model(epsilon_map, sigma_map)
    print(out.shape)


    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    DEVICE = "cuda:2"
    N_train = [2500]
    N_test_start = [2500]
    N_test = [250]
    N_class = len(N_train)
    data_sizeX = 200
    data_sizeY = 600
    mask_sizeX = 256
    mask_sizeY = 256
    num_channel = 1
    train_data1 = []
    train_data2 = []
    train_mask= []
    test_data1 = []
    test_data2 = []
    test_mask = []
    path = 'munnezza_data/' 
    sub_path = 'exp/'

    # Load data
    for i in range(N_class):
        for j in range(N_train[i]):
            data1 = sio.loadmat(path+'dataset/%d/data1/%d.mat'%(i+1, j+1))['data1']
            train_data1.append(data1.reshape((data_sizeX,data_sizeY,1)))
            data2 = sio.loadmat(path+'dataset/%d/data2/%d.mat'%(i+1, j+1))['data2']
            train_data2.append(data2.reshape((data_sizeX,data_sizeY,1)))
            mask = sio.loadmat(path+'dataset/%d/mask/%d.mat'%(i+1, j+1))['mask']
            train_mask.append(mask.reshape((mask_sizeX,mask_sizeY,1)))
        
        for j in range(N_test_start[i], N_test_start[i]+N_test[i]):
            data1 = sio.loadmat(path+'dataset/%d/data1/%d.mat'%(i+1, j+1))['data1']
            test_data1.append(data1.reshape((data_sizeX,data_sizeY,1)))
            data2 = sio.loadmat(path+'dataset/%d/data2/%d.mat'%(i+1, j+1))['data2']
            test_data2.append(data2.reshape((data_sizeX,data_sizeY,1)))
            mask = sio.loadmat(path+'dataset/%d/mask/%d.mat'%(i+1, j+1))['mask']
            test_mask.append(mask.reshape((mask_sizeX,mask_sizeY,1)))

    train_data1 = torch.from_numpy(np.array(train_data1)).permute(0, 3, 1, 2)
    train_data2 = torch.from_numpy(np.array(train_data2)).permute(0, 3, 1, 2)
    train_mask = torch.from_numpy(np.array(train_mask)).permute(0, 3, 1, 2)
    test_data1 = torch.from_numpy(np.array(test_data1)).permute(0, 3, 1, 2)
    test_data2 = torch.from_numpy(np.array(test_data2)).permute(0, 3, 1, 2)
    test_mask = torch.from_numpy(np.array(test_mask)).permute(0, 3, 1, 2)

    print(train_data1.dtype)

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    class SimpleDataset(Dataset):
        def __init__(self, epsilon:torch.Tensor, sigma:torch.Tensor, mask:torch.Tensor) -> None:
            super().__init__()
            self.epsilon_map = epsilon
            self.sigma_map = sigma
            self.label = mask
            assert len(epsilon) == len(sigma) == len(mask)

        def __len__(self):
            return len(self.label)
        
        def __getitem__(self, index):
            return self.epsilon_map[index], self.sigma_map[index], self.label[index]

    train_dataset = SimpleDataset(train_data1, train_data2, train_mask)
    test_dataset = SimpleDataset(test_data1, test_data2, test_mask)

    train_loader = DataLoader(train_dataset, 10, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, 10, shuffle=False, num_workers=16)

    EPOCHS = 100
    train_losses = []
    for i in tqdm(range(EPOCHS)):
        for e, s, label in train_loader:
            e = e.to(DEVICE)
            s = s.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            pred = model(e, s)
            l = loss_fn(pred, label)
            l.backward()
            train_losses.append(float(l.item()))
            optimizer.step()
        
        # test loss
        test_loss = []
        with torch.no_grad():
            for e, s, label in test_loader:
                e = e.to(DEVICE)
                s = s.to(DEVICE)
                label = label.to(DEVICE)
                pred = model(e, s)
                l = loss_fn(pred, label)
                test_loss.append(l.item())
            print(f"End of epoch {i}: test loss:", np.asarray(test_loss).mean())
        
    
    test_loss = []
    with torch.no_grad():
        for e, s, label in test_loader:
            e = e.to(DEVICE)
            s = s.to(DEVICE)
            label = label.to(DEVICE)
            pred = model(e, s)
            l = loss_fn(pred, label)
            test_loss.append(l.item())

    print("test_loss:", test_loss)
    plt.plot(train_losses)
    plt.show()


