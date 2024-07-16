import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2,modes1 = None, modes2 = None):
        super(SpectralConv2d_Uno, self).__init__()

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1 
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 
            self.modes2 = dim2//2 
        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm = 'forward')

        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2),norm = 'forward')
        return x


class pointwise_op_2D(nn.Module):
    def __init__(self, in_codim, out_codim,dim1, dim2):
        super(pointwise_op_2D,self).__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)

        #ft = torch.fft.rfft2(x_out)
        #ft_u = torch.zeros_like(ft)
        #ft_u[:dim1//2-1,:dim2//2-1] = ft[:dim1//2-1,:dim2//2-1]
        #ft_u[-(dim1//2-1):,:dim2//2-1] = ft[-(dim1//2-1):,:dim2//2-1]
        #x_out = torch.fft.irfft2(ft_u)
        
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True, antialias=True)
        return x_out


class OperatorBlock_2D(nn.Module):
    def __init__(self, in_codim, out_codim,dim1, dim2,modes1,modes2, Normalize = False, Non_Lin = True):
        super(OperatorBlock_2D,self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1,dim2,modes1,modes2)
        self.w = pointwise_op_2D(in_codim, out_codim, dim1,dim2)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim),affine=True)

    def forward(self,x, dim1 = None, dim2 = None):
        x1_out = self.conv(x,dim1,dim2)
        x2_out = self.w(x,dim1,dim2)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class Network(nn.Module):
    def __init__(self, in_width, width, pad = 0, **kwargs):
        super(Network, self).__init__()
        self.in_width = in_width
        self.width = width 
        
        self.padding = pad

        self.fc = nn.Linear(self.in_width + 4, self.width)

        self.L0 = OperatorBlock_2D(self.width, self.width//2, 256, 256, 32, 32)
        self.L1 = OperatorBlock_2D(self.width//2, self.width//4, 128, 128, 16, 16)
        self.L2 = OperatorBlock_2D(self.width//4, self.width//8, 64, 64, 8, 8)
        self.L3 = OperatorBlock_2D(self.width//8, self.width//16, 32, 32, 4, 4)

        self.L4 = OperatorBlock_2D(self.width//16, self.width//8, 64, 64, 8, 8)
        self.L5 = OperatorBlock_2D(self.width//4, self.width//4, 128, 128, 16, 16)
        self.L6 = OperatorBlock_2D(self.width//2, self.width//2, 256, 256, 32, 32)
        self.L7 = OperatorBlock_2D(self.width, self.width, 512, 512, 64, 64)

        self.fc1 = nn.Linear(self.width, 1)
        
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((torch.sin(gridx),torch.sin(gridy),torch.cos(gridx),torch.cos(gridy)), dim=-1).to(device)

    def forward(self, x, underground_data):
        x = F.interpolate(x, size=(512, 512), mode='bicubic', align_corners=True)
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x_fc = self.fc(x)
        x_fc = x_fc.permute(0, 3, 1, 2)
        x_fc = F.pad(x_fc, [self.padding, self.padding, self.padding, self.padding])
        
        D1, D2 = x_fc.shape[-2], x_fc.shape[-1]
        x_c0 = self.L0(x_fc, D1//2, D2//2)
        x_c1 = self.L1(x_c0, D1//4, D2//4)
        x_c2 = self.L2(x_c1, D1//8, D2//8)
        x_c3 = self.L3(x_c2, D1//16, D2//16)

        x_c4 = self.L4(x_c3, D1//8, D2//8)
        x_c4 = torch.cat([x_c4, x_c2], dim=1)
        x_c5 = self.L5(x_c4, D1//4, D2//4)
        x_c5 = torch.cat([x_c5, x_c1], dim=1)
        x_c6 = self.L6(x_c5, D1//2, D2//2)
        x_c6 = torch.cat([x_c6, x_c0], dim=1)
        x_c7 = self.L7(x_c6, D1, D2)
        if self.padding != 0:
            x_c7 = x_c7[..., self.padding:-self.padding, self.padding:-self.padding]
        x_c7 = x_c7.permute(0, 2, 3, 1)

        x_out = self.fc1(x_c7)
        x_out = x_out.permute(0, 3, 1, 2)

        return x_out
