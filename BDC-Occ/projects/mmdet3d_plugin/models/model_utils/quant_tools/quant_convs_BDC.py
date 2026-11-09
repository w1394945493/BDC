import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import os
from torch.utils.checkpoint import checkpoint
from pdb import set_trace as stx
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ['cuda_visible_device']='2'
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
from .function import BinaryConv2dFunction

# --------------------------------------------- Binarized Basic Units -----------------------------------------------------------------
def binaryconv3x3(in_planes, out_planes, stride=1,groups=1, bias=False):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)


def binaryconv1x1(in_planes, out_planes, stride=1,groups=1, bias=False):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, groups=groups, bias=bias)


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        # stx()
        out = x + self.bias.expand_as(x)
        return out

#*-------------------------------------------------------------#
# 特征重分布（ReDistribution）：通过通道级可学习缩放系数 \(k\) 和偏移量 \(b\)，对特征进行仿射变换 \(y=kx+b\)，
# 自适应调整特征分布及二值化阈值，缓解固定零阈值导致的特征信息损失。
#* 为什么 BNN 需要特征重分布？二值网络使用 sign() 进行激活二值化，而 sign() 对输入的正负号非常敏感。
# 当 \(k>0\) 时，相当于将二值化阈值从 0 调整为：x=-b/k，因此，ReDistribution 可以帮助网络学习更合适的通道级二值化阈值，减少固定零阈值造成的信息损失。
class ReDistribution(nn.Module):
    def __init__(self, out_chn):
        super(ReDistribution, self).__init__()
        self.b = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        self.k = nn.Parameter(torch.ones(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x * self.k.expand_as(x) + self.b.expand_as(x)
        return out

# RPReLU（ReAct PReLU）激活函数，是二值神经网络（BNN）中用于增强特征表达能力的一种设计，典型地应用于 ReActNet。
# 它在普通 PReLU 的前后分别引入可学习偏置，使激活函数不仅能调整负半轴的斜率，还能自适应调整激活阈值和输出分布，以缓解二值化造成的特征信息损失。
class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        # 设置两个可学习偏置的原因：两个偏置的作用不同，不能简单合并。第一个偏置 pr_bias0：调整激活阈值。第二个偏置 pr_bias1：调整输出分布。
        # 它不改变 PReLU 的分段位置，而是整体平移输出特征，调整输出的数值分布。
        self.pr_bias0 = LearnableBias(inplanes)
        # BNN 需要RPReLU的原因：BNN 的激活值通常被压缩为 ±1，容易丢失幅值信息，导致特征表达能力下降。RPReLU 提供三个可学习自由度：
        # 参数	作用
        # 输入偏置 b0	调整 PReLU 的分段阈值
        # 负半轴斜率 a	调整负值特征的响应强度
        # 输出偏置 b1	调整输出特征分布
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)
    # RPReLU 激活函数：在 PReLU 前后分别引入通道级可学习偏置，前者调整激活函数的分段阈值，后者调整输出特征分布，
    # 并通过可学习负半轴斜率增强非线性表达能力，从而缓解二值化造成的特征信息损失。
    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))      #为什么要反复设置可学习偏置？
        return x

#*-------------------------------------------------------------#
# 基于双曲正切函数（tanh）的直通估计器（STE），解决二值激活函数 sign 不可微、梯度几乎处处为零而难以训练的问题。
# 具体而言，前向传播通过 sign 函数将连续特征二值化为离散值；反向传播则利用 detach() 阻断 sign 分支及部分 tanh 分支的梯度，
# 使梯度仅通过可微的 tanh 分支传播，从而以 tanh 的导数近似 sign 的梯度。
# 引入可学习参数 β 自适应调节 tanh 的形状及梯度分布，实现前向二值化与反向可微优化的兼顾，使二值神经网络能够通过梯度下降进行训练。
class Spectral_Binary_Activation(nn.Module):
    def __init__(self):
        super(Spectral_Binary_Activation, self).__init__()
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        #* 基于直通估计器（Straight-Through Estimator, STE）的二值激活函数，用于二值神经网络
        #* 前向传播使用 sign() 实现真正的二值化，反向传播则借助 tanh() 的梯度，使网络能够正常训练。

        # 普通二值函数：能把连续特征变成离散值，但存在一个严重问题 sign() 几乎处处梯度为 0。
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(x),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # scaling_factor = scaling_factor.detach()

        binary_activation_no_grad = torch.sign(x)
        tanh_activation = torch.tanh(x*self.beta)
        #*-----------------------------------------------------------------------#
        #* detach() 的作用是将 tensor 从计算图中分离出来，使其不参与梯度计算。这里的设计是为了在前向传播中使用二值化的输出，
        #* 而在反向传播中使用 tanh 的梯度，从而实现梯度的传递。
        out = binary_activation_no_grad.detach() - tanh_activation.detach() + tanh_activation  #.detach() 不需要计算其梯度，不具有梯度grad，此处为何是这样设计BinaryActivation?

        return out

# 带权重缩放因子的二值卷积模块（HardBinaryConv），主要实现卷积权重的二值化，并通过直通估计器（STE）解决二值权重难以反向传播的问题。
class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        #* (1) 计算权重缩放因子，该操作计算每个输出通道对应卷积核权重的绝对值均值：每个输出通道具有独立的缩放因子，用于补偿直接将浮点权重量化为 ±1 后造成的幅值信息损失。
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach() # 表示不通过缩放因子的计算路径传播梯度。
        #* (2) 权重二值化：使用 torch.sign() 将实数权重二值化为 ±1，得到 binary_weights_no_grad。由于 sign() 几乎处处不可导，因此在反向传播时不通过该分支计算梯度。
        # stx()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)   #用torch.sign()函数进行二值化
        #* (3) 通过 STE 实现反向传播
        # 与前面 tanh STE 的思想相同，前向传播保留二值权重，反向传播则利用 clamp() 的梯度。
        # stx()
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #*------------------------------------------------------------------------#
        #* (4) 使用二值权重进行卷积操作:当前代码虽然实现了权重数值二值化，但仍然调用 PyTorch 的普通 F.conv2d，并没有直接实现 XNOR、BitCount 等二值位运算加速。
        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

        return y

    #* 现有自定义方法，并不会减少用时
    # def forward(self, x):
    #     return BinaryConv2dFunction.apply(
    #         x,
    #         self.weight,
    #         self.bias,
    #         self.stride,
    #         self.padding,
    #         self.dilation,
    #         self.groups
    #     )
#*-------------------------------------------------------------#
#* 旧版本SEBlock
# 二值化的 SE 注意力模块（Squeeze-and-Excitation Block），主要用于学习不同特征通道的重要性，并通过通道级权重对输入特征进行自适应增强或抑制。
# 与传统 SE Block 不同，BDC-Occ 这里使用了 binaryconv1x1 替代普通全精度卷积，以降低注意力模块的计算开销。
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, with_cp = False):
        super(SEBlock, self).__init__()
        self.with_cp = with_cp
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            binaryconv1x1(in_planes=in_channels, out_planes=in_channels//reduction, stride=1, bias=False),
            RPReLU(in_channels),
            binaryconv1x1(in_planes=in_channels//reduction, out_planes=in_channels, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        if self.with_cp:
            y = checkpoint(self.fc, y)
        else:
            y = self.fc(y)
        return x * y

#*============================================================#
#* 作者提供的训练权重中的SEBlock还是上述旧版本
# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16, with_cp = False):
#         super(SEBlock, self).__init__()
#         self.with_cp = with_cp
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.with_bn = False
#         self.fc = nn.Sequential(
#             binaryconv1x1(in_planes=in_channels, out_planes=in_channels//reduction, stride=1, bias=False),
#             RPReLU(in_channels),
#             binaryconv1x1(in_planes=in_channels//reduction, out_planes=in_channels, stride=1, bias=False),
#             nn.Sigmoid()
#         )
#         self.fc2 = nn.Sequential(
#             binaryconv1x1(in_planes=in_channels, out_planes=in_channels//reduction, stride=1, bias=False),
#             RPReLU(in_channels),
#             binaryconv1x1(in_planes=in_channels//reduction, out_planes=in_channels, stride=1, bias=False),
#             nn.Tanh()
#         )
#     def forward(self, x):
#         b, c, h, w = x.size()
#         y = self.avg_pool(x)
#         if self.with_cp:
#             scale_x = checkpoint(self.fc, y)
#             bias_x = checkpoint(self.fc2, y)
#         else:
#             scale_x = self.fc(y)
#             bias_x = self.fc2(y)

#         return scale_x * x + bias_x


# --------------------------------------------- Binarized Conv Units -----------------------------------------------------------------

class BinaryConv2dBDC(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC, self).__init__()
        self.with_bn = with_bn
        self.with_cp = with_cp
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.move0 = ReDistribution(in_channels)
        self.binary_activation = Spectral_Binary_Activation()
        self.binary_conv_relu = nn.Sequential(
            binaryconv3x3(in_planes=in_channels, out_planes=in_channels, stride=stride, bias=bias, groups=groups),
            nn.BatchNorm2d(in_channels) if self.with_bn else nn.Identity(),
            RPReLU(in_channels),
            binaryconv1x1(in_planes=in_channels, out_planes=out_channels, stride=1, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels) if self.with_bn else nn.Identity(),
            RPReLU(out_channels),
        )

        # SE layer
        self.se_block = SEBlock(out_channels, reduction=1, with_cp=with_cp)

        if self.stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        if self.with_bn:
            self.bn_fp = nn.BatchNorm2d(out_channels)

    def forward(self, x): # (1 256 200 200)
        if self.stride == 2:
            identity = self.downsample(x)
        else:
            identity = x
        if self.out_channels == self.in_channels * 2:
            identity = torch.cat([identity, identity], dim=1)
        #* (1) ReDistribution: 特征重分布：可学习的通道级仿射变换模块，通过可学习的缩放和平移参数，调整各通道特征的数值分布。
        out = self.move0(x)
        #* (2) Binary Activation: 二值激活函数，使用 sign() 将特征二值化，同时通过 tanh() 的梯度实现反向传播的可微性。
        out = self.binary_activation(out) # 二值化，全为-1或1
        #* (3) Binary Conv + RPReLU: 二值卷积模块，使用二值化权重进行卷积操作，并通过 RPReLU 激活函数增强特征表达能力。
        if self.with_cp:
            out = checkpoint(self.binary_conv_relu, out)
        else:
            out = self.binary_conv_relu(out)
        #* (4) SE Block: Squeeze-and-Excitation 注意力模块，学习各通道的重要性，并通过通道级权重对特征进行自适应增强或抑制。
        out = self.se_block(out)
        if self.with_bn:
            out = out + self.bn_fp(identity) # batchnorm 之后再加残差
        else:
            out = out + identity
        return out



class BinaryConv2dBDC_Down(nn.Module):
    '''
    降采样且通道数翻倍
    input: b,c,h,w
    output: b,2c,h/2,w/2
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Down, self).__init__()
        self.with_cp = with_cp
        self.biconv_1 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''
        out = self.avg_pool(x)
        out_1 = out
        out_2 = out_1.clone()
        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        out = torch.cat([out_1, out_2], dim=1)

        return out


class BinaryConv2dBDC_Up(nn.Module):
    '''
    上采样且通道数不变
    input: b,c,h,w
    output: b,c,2h,2w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Up, self).__init__()
        self.with_cp=with_cp
        bi_channels = out_channels // 2
        self.biconv_1 = BinaryConv2dBDC(bi_channels, bi_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(bi_channels, bi_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        # self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c,2h,2w
        '''
        b,c,h,w = x.shape
        out = F.interpolate(x, scale_factor=2, mode='bilinear')

        out_1 = out[:,:c//2,:,:]
        out_2 = out[:,c//2:,:,:]

        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)

        out = torch.cat([out_1, out_2], dim=1)  # output: b,c,2h,2w
        # out = (out_1 + out_2) / 2 # output: b,c/2,2h,2w

        return out


class BinaryConv2dBDC_Fusion_Decrease(nn.Module):
    '''
    空间尺寸不变且通道数减半 - Upsample 去掉上采样
    input: b,c,h,w
    output: b,c/2,h,w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Fusion_Decrease, self).__init__()
        self.with_cp=with_cp
        self.biconv_1 = BinaryConv2dBDC(out_channels, out_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(out_channels, out_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        # self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,h,w
        '''
        b,c,h,w = x.shape
        out = x

        out_1 = out[:,:c//2,:,:]
        out_2 = out[:,c//2:,:,:]

        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)

        out = (out_1 + out_2) / 2

        return out


class BinaryConv2dBDC_Fusion_Increase(nn.Module):
    '''
    空间尺寸不变且通道数翻倍 - Downsample 去掉下采样
    input: b,c,h,w
    output: b,2c,h,w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Fusion_Increase, self).__init__()
        self.with_cp=with_cp
        self.biconv_1 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        # self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''
        # stx()
        out_1 = x
        out_2 = out_1.clone()
        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        out = torch.cat([out_1, out_2], dim=1)

        return out


class BinaryConv2dBDC_Down_Constant(nn.Module):
    '''
    降采样且通道数翻倍
    input: b,c,h,w
    output: b,c,h/2,w/2
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Down_Constant, self).__init__()
        self.with_cp=with_cp
        self.biconv_1 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''
        out = self.avg_pool(x)
        out_1 = out
        out_2 = out_1.clone()
        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        out = out_1 + out_2

        return out




# --------------------------------------------- Binarized Block Units -----------------------------------------------------------------

class BasicBinaryBlockBDC(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None, with_bn=False, with_cp=False):
        super(BasicBinaryBlockBDC, self).__init__()
        self.conv1 = BinaryConv2dBDC(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            with_bn=with_bn,
            with_cp=with_cp
            )
        self.conv2 = BinaryConv2dBDC(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            with_bn=with_bn,
            with_cp=with_cp
            )
        self.downsample = downsample
        self.relu = RPReLU(channels_out)
        self.with_cp=with_cp

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)


class DownsampleBinaryBlockBDC(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None, with_bn=False, with_cp=False):
        super(DownsampleBinaryBlockBDC, self).__init__()
        self.conv1 = BinaryConv2dBDC_Down(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            with_bn=with_bn,
            with_cp=with_cp
            )
        self.conv2 = BinaryConv2dBDC(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            with_bn=with_bn,
            with_cp=with_cp
            )
        self.downsample = downsample
        self.relu = RPReLU(channels_out)
        self.with_cp=with_cp

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)
