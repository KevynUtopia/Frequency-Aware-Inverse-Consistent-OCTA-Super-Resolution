from tkinter import NE

from cv2 import namedWindow
from pytorch_wavelets import DWTForward
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils import high_pass, low_pass


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class phase_consistency_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        radius = 5
        rows, cols = x[0][0].shape
        center = int(rows / 2), int(cols / 2)

        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
                mask[i, j] = 1 - np.exp(-0.5 *  distance_u_v / (radius ** 2))
        m = torch.from_numpy(mask).float().cuda()

        f_x = torch.fft.fft2(x[0])
        fshift_x = torch.fft.fftshift(f_x)
        amp_x = (m * torch.log(torch.abs(fshift_x))).flatten()
        f_y = torch.fft.fft2(y[0])
        fshift_y = torch.fft.fftshift(f_y)
        amp_y = (m * torch.log(torch.abs(fshift_y))).flatten()
        return -torch.cosine_similarity(amp_x, amp_y, dim=0)

# Loss functions
class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        use_bias = True
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

#################################################################
################# Frequency Discriminator ####################
#################################################################
class FS_DiscriminatorA(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, wgan=False, highpass=True, D_arch='FSD',
                 norm_layer='Instance', filter_type='gau', cs='sum'):
        super(FS_DiscriminatorA, self).__init__()

        self.wgan = wgan
        n_input_channel = 1
        
        self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
        self.filter = self.filter_wavelet
        self.cs = cs

        print('# FS type: {}, kernel size={}'.format(filter_type.lower(), kernel_size))
        

        self.net = Discriminator(input_nc=1)
        if cs=='sum':
          self.net_dwt = Discriminator(input_nc=1)
        else:
          self.net_dwt = Discriminator(input_nc=3)
        self.out_net = nn.Softmax()

    def forward(self, x, y=None):
        dwt, ximg = self.filter(x)
        # LL, LH, HL, HH, ximg = self.filter(x)
        x = self.net(ximg)
        x_D = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        dwt_D = self.net_dwt(dwt)
        dwt_D = F.avg_pool2d(dwt_D, dwt_D.size()[2:]).view(x.size()[0], -1)

        # return (torch.flatten(x_D))
        return (torch.flatten(0.7*x_D + 0.3*dwt_D))

    def filter_wavelet(self, x, norm=True):
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
        if norm:
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        if self.cs.lower() == 'sum':
            return LL, x

        elif self.cs.lower() == 'each':
            return LL, LH, HL, HH, x
        elif self.cs.lower() == 'cat':
            return torch.cat((LH, HL, HH), 1), x
        else:
            raise NotImplementedError('Wavelet format [{:s}] not recognized'.format(self.cs))


class FS_DiscriminatorB(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, wgan=False, highpass=True, D_arch='FSD',
                 norm_layer='Instance', filter_type='gau', cs='cat'):
        super(FS_DiscriminatorB, self).__init__()

        self.wgan = wgan
        n_input_channel = 1
        
        self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
        self.filter = self.filter_wavelet
        self.cs = cs
        # n_input_channel = 3
        n_input_channel = 1

        print('# FS type: {}, kernel size={}'.format(filter_type.lower(), kernel_size))
        

        self.net = Discriminator(input_nc=1)
        if cs=='sum':
          self.net_dwt = Discriminator(input_nc=1)
        else:
          self.net_dwt = Discriminator(input_nc=3)
        self.out_net = nn.Softmax()


    def forward(self, x, y=None):
        dwt, ximg = self.filter(x)
        # LL, LH, HL, HH, ximg = self.filter(x)
        x = self.net(ximg)
        x_D = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        dwt_D = self.net_dwt(dwt)
        dwt_D = F.avg_pool2d(dwt_D, dwt_D.size()[2:]).view(x.size()[0], -1)

        # return dwt_D
        return (torch.flatten(0.7*x_D + 0.3*dwt_D))
        # return (torch.flatten(x_D))



    def filter_wavelet(self, x, norm=True):
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
        if norm:
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        if self.cs.lower() == 'sum':
            return HH, x

        elif self.cs.lower() == 'each':
            return LL, LH, HL, HH, x
        elif self.cs.lower() == 'cat':
            return torch.cat((LH, HL, HH), 1), x
        else:
            raise NotImplementedError('Wavelet format [{:s}] not recognized'.format(self.cs))


class NetworkA2B(nn.Module):
    def __init__(self, use_bias=False):
        super(NetworkA2B, self).__init__()
        self.unet = UnetGenerator(input_nc=64, output_nc=64, num_downs=7)
        self.shallow_frequency = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.LeakyReLU(0.2, True),nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(128),
                                    nn.ReLU(True),nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(64)

                                      ])
        # self.shallow_frequency = shallowNet(in_dim=1, out_dim=64, up=False)                         
        self.shallow_up = shallowNet(up=True)
        self.skip = nn.Sequential(*[nn.ReLU(True),
                                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                        nn.BatchNorm2d(64)
                                      ])
    
        self.unet_up = nn.Sequential(*[nn.ReLU(True),
                                        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                        nn.BatchNorm2d(64)
                                      ])
        self.A2B_input = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=use_bias)
                                      ])
        self.resnet = ResnetGenerator(input_nc=64, output_nc=64, n_blocks=8)
    
    def forward(self, lf, hf):
        lf_feature = self.shallow_frequency(lf) #64x128^2
        hf_feature_input = self.A2B_input(hf) #64x128^2

        hf_feature = self.skip(torch.cat([hf_feature_input, self.resnet(hf_feature_input)], 1)) #64*256^2
        # return None, None, feature_map
        return lf_feature, hf_feature, self.shallow_up(torch.cat([lf_feature, hf_feature], 1))


class NetworkB2A(nn.Module):
    def __init__(self, use_bias=False):
        super(NetworkB2A, self).__init__()
        # self.unet = UnetGenerator(input_nc=1, output_nc=1, num_downs=8)
        self.shallow_frequency = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.LeakyReLU(0.2, True),nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(128),
                                    nn.ReLU(True),nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(64)

                                      ])
        self.shallow_up = shallowNet(up=True)
        self.skip = nn.Sequential(*[nn.ReLU(True),
                                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                        nn.BatchNorm2d(64)
                                      ])
        self.resnet = ResnetGenerator(input_nc=128, output_nc=64, n_blocks=8)
        self.B2A_input = nn.Sequential(*[nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1, bias=use_bias)
                                      ])

    
    def forward(self, hf, lf):
        hf_feature = self.shallow_frequency(hf) #64x256^2
        # feature_map = self.unet(lf) #128x128^2
        lf_feature = self.resnet(self.B2A_input(lf)) #64x256^2
        # lf_feature = self.skip(feature_map) #64*128^2
        

        # return None, None, feature_map
        return hf_feature, lf_feature, self.shallow_up(torch.cat([hf_feature, lf_feature], 1))



class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=1, output_nc=1, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, B):
        """Standard forward"""
        # hf = high_pass(input[0], i=5).unsqueeze(0).unsqueeze(0) # (1, 320) 5
        # input = (hf+input)/2.0
        return self.model(B)



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=True):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = True
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] #+ up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            out = self.model(x)
            return out
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_bias=False)

    def build_conv_block(self, dim=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        # nn.ReplicationPad2d(1)
        conv_block = [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(dim), 
                       nn.ReLU(True),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class shallowNet(nn.Module):
    def __init__(self, in_dim = 128, out_dim=1, up=False):
        super(shallowNet, self).__init__()
        # if A2B:
        #       model = [nn.ReLU(True), nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64)]
        # else:
        #   model = [nn.ReLU(True), nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)]
        if up:
            model = [nn.ReLU(True), nn.ConvTranspose2d(in_dim, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64)]
        else:
            model = [nn.ReLU(True), nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)]
        model += [ResnetBlock()]
        model += [ResnetBlock()]
        model += [ResnetBlock()]
        # model += [ResnetBlock()]
        model += [nn.ReLU(True), nn.Conv2d(64, out_dim, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=64, output_nc=64, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=8, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResidualBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResidualBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
