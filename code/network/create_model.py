import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *
import cv2
from einops import rearrange

ngf = 32
layer_specs = [
    ngf,
    ngf * 2,
    ngf * 4,
    ngf * 8
]
class Generator(nn.Module):
    def __init__(self, img_size=256):
        super(Generator, self).__init__()

        self.enconv1 = nn.Sequential(
            ConvGroup(3, layer_specs[1], 4, 2, 2),
            ConvGroup(layer_specs[1], layer_specs[1], 4, 2, 2),
            ConvGroup(layer_specs[1], layer_specs[1], 4, 2, 2),
        )
        self.enconv2 = nn.Sequential(
            ConvGroup(3, layer_specs[1], 4, 2, 2),
            ConvGroup(layer_specs[1], layer_specs[1], 4, 2, 2),
            ConvGroup(layer_specs[1], layer_specs[1], 4, 2, 2),
        )
        self.enconv2_2 = nn.Sequential(
            ConvGroup(1, layer_specs[1], 4, 2, 2),
            ConvGroup(layer_specs[1], layer_specs[1], 4, 2, 2),
            ConvGroup(layer_specs[1], layer_specs[1], 4, 2, 2),
        )
        self.enconv3 = nn.Sequential(
            ConvGroup(layer_specs[1]*2, layer_specs[2], 3, 1, 1),
            ConvGroup(layer_specs[2], layer_specs[2], 4, 2, 1),
            ConvGroup(layer_specs[2], layer_specs[2], 4, 2, 2),
        )
        self.enconv4 = nn.Sequential(
            ConvGroup(layer_specs[1], layer_specs[2], 4, 2, 2),
            ConvGroup(layer_specs[2], layer_specs[2], 3, 1, 2),
            ConvGroup(layer_specs[2], layer_specs[2], 4, 2, 2),
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(layer_specs[2], layer_specs[2], bias=True),
            nn.ReLU(),
            nn.Linear(layer_specs[2], layer_specs[2], bias=True),
            nn.ReLU(),
            nn.Linear(layer_specs[2], 3, bias=True),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(layer_specs[2] + 3, layer_specs[2], bias=True),
            nn.ReLU(),
            nn.Linear(layer_specs[2], layer_specs[2], bias=True),
            nn.ReLU(),
            nn.Linear(layer_specs[2], layer_specs[2], bias=True),
            nn.ReLU(),
            nn.Linear(layer_specs[2], 4, bias=True),
            nn.Sigmoid()
        )

    
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        # inputs = torch.squeeze(inputs, dim=1)
        inputs = inputs.permute(0,1, 4,2,3)
        mask = inputs[:, -3:-2,:,:,:]
        albedo = inputs[:, -2:-1,:,:,:]
        bias = inputs[:, -1,:,:,:]
        bias = torch.sum(bias, dim=1, keepdim=True)
        mask = torch.mean(mask, dim=2, keepdim=True)
        albedo = torch.squeeze(albedo, dim=1)
        inputs = inputs[:,:-3,:,:,:]
        # one = torch.ones_like(bias)
        # zero = torch.zeros_like(bias)
        # bias = torch.where(bias!=0, one, zero)
        # inputs = inputs*mask
        inputs = rearrange(inputs, 'n d c h w -> (n d) c h w')
        # inputs = torch.matmul(inputs, bias)

        x = self.enconv1(inputs)
        albedo_feature = self.enconv2(albedo)
        bias_feature = self.enconv2_2(bias)
        x = rearrange(x, '(n d) c h w -> n d c h w', n = batch_size)
        x, _ = torch.max(x, dim=1, keepdim=False)
        print(albedo_feature.shape)
        print(bias_feature.shape)
        x1 = torch.cat((x, albedo_feature), 1)
        x1 = self.enconv3(x1)
        print(x1.shape)
        x1 = self.max_pool(x1)
        x1 = rearrange(x1, 'n c a b -> n (c a b)')
        rgb = self.mlp(x1)

        x2 = x*bias_feature
        x2 = self.enconv4(x2)
        x2 = self.max_pool(x2)
        x2 = rearrange(x2, 'n c a b -> n (c a b)')
        x2 = torch.cat((x2, rgb), 1)
        sigma = self.mlp2(x2)
        g = sigma[:,-1:]
        sigma = sigma[:,:-1]

        return torch.cat((sigma, rgb, g), 1)

