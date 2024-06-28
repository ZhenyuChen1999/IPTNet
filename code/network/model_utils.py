import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations
import cv2

class PaddingModule(nn.Module):
    def __init__(self, padding = 0):
        super(PaddingModule, self).__init__()
        self.pad = padding
    def forward(self, inputs):
        return F.pad(inputs, [self.pad, self.pad, self.pad, self.pad], "reflect")
        #return F.pad(inputs, [0, 1, 0, 1])

class FeatureConcat(nn.Module):
    def __init__(self, input_channel, output_channel, bias=True):
        super(FeatureConcat, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel*2, output_channel, kernel_size=1, stride=1, bias=bias, padding=0),
            nn.LeakyReLU(0.2)
            )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)
        return x

class FeatureConcatUp(nn.Module):
    def __init__(self, input_channel, output_channel, bias=False):
        super(FeatureConcatUp, self).__init__()
        self.concat = FeatureConcat(input_channel, output_channel, bias=bias)
        self.conv = Upsample(output_channel, output_channel)

    def forward(self, x1, x2):
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x

class ConvGroup(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, blocks=1):
        super(ConvGroup, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module('conv0', BasicConv2D(input_channel, output_channel, kernel_size=kernel_size, stride=stride))
        for i in range(1, blocks):
            self.blocks.add_module('conv{}'.format(i),
                                   BasicConv2D(output_channel, output_channel, kernel_size=3, stride=1))

    def forward(self, inputs):
        return self.blocks(inputs)

class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, useNorm=False, padding=0):
        super(BasicConv2D, self).__init__()
        self.pad = (kernel_size-stride)//2  #1
        self.useNorm = useNorm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        # inputs = F.pad(inputs, [self.pad, self.pad, self.pad, self.pad], "reflect")
        x = self.conv(inputs)
        if self.useNorm:
            x = InstanceNorm(x)
        x = self.act(x)
        return x

class Upsample(nn.Module):
    def __init__(self, input_channel, output_channel, scale_factor=2):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(
            Interpolate(scale_factor),
            BasicConv2D(input_channel, output_channel, kernel_size=3, stride = 1, useNorm=False)
        )

    def forward(self, inputs):
        output = self.conv(inputs)
        return output


class ChannelAttention(nn.Module):
    def __init__(self, input_channel, output_channel, reduction=8, bias=True):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(input_channel, input_channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel // reduction, output_channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        ret = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return ret


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()

        self.conv_1 = Conv_IN_LRelu(channels, channels, kernel_size, stride=1, padding=padding)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding)

    def forward(self, x):
        ret = x + self.conv_2(self.conv_1(x))
        return ret

class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(InceptionBlock, self).__init__()
        assert out_channel % 2 == 0
        self.b1_conv = nn.Conv2d(in_channel, int(out_channel / 2), kernel_size=3, stride=stride, padding=1)
        self.b1_conv_act = nn.LeakyReLU(0.2)

        self.b2_conv1 = nn.Conv2d(in_channel, int(out_channel / 2), kernel_size=3, stride=stride, padding=1)
        self.b2_conv1_act = nn.LeakyReLU(0.2)

        self.b2_conv2 = nn.Conv2d(int(out_channel / 2), int(out_channel / 2), kernel_size=3, stride=1, padding=1)
        self.b2_conv2_act = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        # in_channel
        b1 = self.b1_conv_act(self.b1_conv(inputs))
        # int(out_channel / 2)
        b2 = self.b2_conv1_act(self.b2_conv1(inputs))
        # int(out_channel / 2)
        b2 = self.b2_conv2_act(self.b2_conv2(b2))
        # int(out_channel / 2)
        output = torch.cat([b1, b2], dim=1)
        # out_channel
        return output


class InstanceNorm(nn.Module):
    def __init__(self):
        super(InstanceNorm, self).__init__()
    def forward(self, inputs):
        mean = torch.mean(inputs, dim=[2, 3], keepdim=True)
        variance = torch.var(inputs, dim=[2, 3], keepdim=True)
        # [batchsize ,1,1, channelNb]
        variance_epsilon = 1e-5
        variance = torch.sqrt(variance + variance_epsilon)
        outgf = torch.cat([mean, variance], 1)
        normalized = (inputs - mean) / variance
        return normalized, outgf


class GatedConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, dilation=1):
        super(GatedConv2d, self).__init__()
        self.gated_mask = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.gated_mask_act = nn.Sigmoid()

        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation)
        self.conv_act = nn.LeakyReLU(0.2)

        self.innorm = InstanceNorm()

    def forward(self, inputs):
        # in
        mean = self.gated_mean(inputs)
        # out
        mask = self.gated_mask_act(self.gated_mask(inputs))
        # out
        inputs, mean1 = self.innorm(inputs)
        x = self.conv_act(self.conv(inputs))
        x = x * mask + mean
        return x


class Styleconv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation_rate=1, padding=0):
        super(Styleconv2, self).__init__()
        self.stride = stride
        self.pad = (kernel_size-stride)//2

        self.conv = BasicConv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, useNorm=False, padding=0)
        if in_channels ==3:
            self.fc1 = nn.Conv2d(in_channels*4, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.fc1 = nn.Conv2d(in_channels*3, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.fc3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # self.fc1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.fc2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.fc3 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.norm = InstanceNorm()

    def forward(self, inputs, gf):
        # inputs = F.pad(inputs, [self.pad, self.pad, self.pad, self.pad], "reflect")
        x, gf1 = self.norm(inputs)
        gf = torch.cat([gf, gf1], 1)
        gf = self.fc1(gf)
        x = self.conv(x)
        gf = self.act(gf)
        mean = self.fc2(gf)
        variance = self.fc3(gf)
        x = x * variance + mean
        return x, gf

from einops import rearrange

#Global_perception Self Attention
class GPSA(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, channel, patchnum):
        super(GPSA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, 128, bias=True),
            nn.ReLU()
        )
        self.conv2 = nn.Linear(in_channels, 64)

    def forward(self, inputs):#(b, h, w, c)
        #y = rearrange(inputs, 'b h w c -> b c (h w)')
        y = self.conv1(inputs)
        y = self.conv2(y)
        #y = rearrange(y, 'b c (h w) -> b h w c', h=8)
        y = F.interpolate(y, size=[inputs.shape[2], inputs.shape[3]], mode='bilinear', align_corners=True)
        y += inputs
        return y

class create_2conv(nn.Module):
    def __init__(self, channel):
        super(create_2conv, self).__init__()

        self.conv = nn.Sequential(
            BasicConv2D(channel[0], channel[1], 3, 1, useNorm=False),
             nn.LeakyReLU(0.2),
            BasicConv2D(channel[1], channel[2], 3, 1, useNorm=False))

    def forward(self, inputs):
        output = self.conv(inputs)
        return output