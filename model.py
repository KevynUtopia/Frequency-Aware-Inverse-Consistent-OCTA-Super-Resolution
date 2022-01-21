from pytorch_wavelets import DWTForward
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils import high_pass, low_pass

class phase_consistency_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        # print(x.size())
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
        # print(y.size())
        f_y = torch.fft.fft2(y[0])
        fshift_y = torch.fft.fftshift(f_y)
        amp_y = (m * torch.log(torch.abs(fshift_y))).flatten()
        # print(amp_x.size(), amp_y.size())
        return -torch.cosine_similarity(amp_x, amp_y, dim=0)


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
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
        self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128, 64 , 1, 1),
                nn.Conv2d(64 , 128, 1, 1),
                nn.Sigmoid())
        self.out_net = nn.Softmax()

    def forward(self, x, y=None):
        dwt, ximg = self.filter(x)
        # LL, LH, HL, HH, ximg = self.filter(x)
        x = self.net(ximg)
        x_D = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        dwt_D = self.net_dwt(dwt)
        dwt_D = F.avg_pool2d(dwt_D, dwt_D.size()[2:]).view(x.size()[0], -1)


        # img_LL = self.net(LL)
        # img_LL = F.avg_pool2d(img_LL, img_LL.size()[2:]).view(x.size()[0], -1)

        # img_LH = self.net(LH)
        # img_LH = F.avg_pool2d(img_LH, img_LH.size()[2:]).view(x.size()[0], -1)

        # img_HL = self.net(HL)
        # img_HL = F.avg_pool2d(img_HL, img_HL.size()[2:]).view(x.size()[0], -1)

        # img_HH = self.net(HH)
        # img_HH = F.avg_pool2d(img_HH, img_HH.size()[2:]).view(x.size()[0], -1)

        # return dwt_D
        return (torch.flatten(0.8*x_D + 0.2*dwt_D))

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

        self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128, 64 , 1, 1),
                nn.Conv2d(64 , 128, 1, 1),
                nn.Sigmoid())
        self.out_net = nn.Softmax()


    def forward(self, x, y=None):
        dwt, ximg = self.filter(x)
        # LL, LH, HL, HH, ximg = self.filter(x)
        x = self.net(ximg)
        x_D = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        dwt_D = self.net_dwt(dwt)
        dwt_D = F.avg_pool2d(dwt_D, dwt_D.size()[2:]).view(x.size()[0], -1)

        # return dwt_D
        return (torch.flatten(0.9*x_D + 0.1*dwt_D))

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

class UnetGeneratorB2A(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGeneratorB2A, self).__init__()

        ####################### UNet #######################
        ######################################################
        use_bias = False
        # (1, 320) -> (64, 160)
        self.down_1 = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        # (64, 160) -> (128, 80)
        self.down_2 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(128)])
        # (128, 80) -> (256, 40)
        self.down_3 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(256)])
        # (256, 40) -> (512, 20)
        self.down_4 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(512)])
        # (512, 20) -> (1024, 10)
        self.down_5 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(1024)])

        # (1024, 10) -> (1024, 5)
        self.down_6 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        
        ######################################################
        # (1024, 5) -> (1024, 10)
        self.up_6 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(1024)])
        # (1024, 10)*2 -> (512, 20)
        self.up_5 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(512)])
        # (512, 20)*2 -> (256, 40)
        self.up_4 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(256)])
        
        # (256, 40)*2 -> (128, 80)
        self.up_3 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(128)])
        # (128, 80)*2 -> (64, 160)
        self.up_2 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(64)])

        ######################################################
        ######################################################
        self.shallow_frequency = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                      nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(64)])
        
        # self.shallow = nn.Sequential(*[nn.ReLU(True), 
        #                             nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #                             nn.Tanh()])
        self.shallow = shallowNet(A2B=False)

    def forward(self, input):
        """Standard forward"""
        hf = high_pass(input[0], i=5).unsqueeze(0).unsqueeze(0) # (1, 320) 5
        hf = (hf+input)/2.0
        lf = low_pass(input[0], i=14).unsqueeze(0).unsqueeze(0) # (1, 320) 14

        hf_input = self.shallow_frequency(hf) # (64, 160)

        down_1 = self.down_1(lf) # (64, 160)
        down_2 = self.down_2(down_1) # (128, 80)
        down_3 = self.down_3(down_2) # (256, 40)
        down_4 = self.down_4(down_3) # (512, 20)
        down_5 = self.down_5(down_4) # (1024, 10)
        down_6 = self.down_6(down_5) # (1024, 5)

        up_6 = self.up_6(down_6) # (1024, 10)
        up_5 = self.up_5(torch.cat([down_5, up_6], 1)) # (512, 20)
        up_4 = self.up_4(torch.cat([down_4, up_5], 1)) # (256, 40)
        up_3 = self.up_3(torch.cat([down_3, up_4], 1)) # (128, 80)
        up_2 = self.up_2(torch.cat([down_2, up_3], 1)) # (64, 160)

        
        return hf_input, up_2, self.shallow(torch.cat([up_2, hf_input], 1)) # B2A: hf_feature, lf_feature, rc
        # return hf_input, up_2, self.shallow(up_2+hf_input) # B2A: hf_feature, lf_feature, rc

class UnetGeneratorA2B(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGeneratorA2B, self).__init__()

        ####################### UNet #######################
        ######################################################
        use_bias = False
        # # (1, 320) -> (64, 160)
        # self.down_1 = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True)])
        # (64, 160) -> (128, 80)
        self.down_1 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(128)])
        # (128, 80) -> (256, 40)
        self.down_2 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(256)])
        # (256, 40) -> (512, 20)
        self.down_3 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(512)])
        # (512, 20) -> (1024, 10)
        self.down_4 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(1024)])

        # (1024, 10) -> (1024, 5)
        self.down_5 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        
        ######################################################
        # (1024, 5) -> (1024, 10)
        self.up_5 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(1024)])
        # (1024, 10)*2 -> (512, 20)
        self.up_4 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(512)])
        # (512, 20)*2 -> (256, 40)
        self.up_3 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(256)])
        
        # (256, 40)*2 -> (128, 80)
        self.up_2 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(128)])
        # (128, 80)*2 -> (64, 160)
        self.up_1 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    nn.BatchNorm2d(64)])

        ######################################################
        ######################################################

        self.shallow_frequency = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                      nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                      nn.BatchNorm2d(64)])
        
        # self.shallow = nn.Sequential(*[nn.ReLU(True), 
        #                             nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=use_bias),
        #                             nn.Tanh()])
        self.shallow = shallowNet(A2B=True)

    def forward(self, input):
        """Standard forward"""
        hf = high_pass(input[0], i=10).unsqueeze(0).unsqueeze(0) # (1, 320) (10 adamw) 16.22 # (gtimg+hp_lr_img)/2.0
        hf = (hf+input)/2.0
        lf = low_pass(input[0], i=8).unsqueeze(0).unsqueeze(0) # (1, 320) (8 adamw)

        lf_input = self.shallow_frequency(lf)

        hf_input = self.shallow_frequency(hf)# (64, 160)
        down_1 = self.down_1(hf_input) # (128, 80)
        down_2 = self.down_2(down_1) # (256, 40)
        down_3 = self.down_3(down_2) # (512, 20)
        down_4 = self.down_4(down_3) # (1024, 10)
        down_5 = self.down_5(down_4) # (1024, 5)

        up_5 = self.up_5(down_5) # (1024, 10)
        up_4 = self.up_4(torch.cat([down_4, up_5], 1)) # (512, 20)
        up_3 = self.up_3(torch.cat([down_3, up_4], 1)) # (256, 40)
        up_2 = self.up_2(torch.cat([down_2, up_3], 1)) # (128, 80)
        up_1 = self.up_1(torch.cat([down_1, up_2], 1)) # (64, 160)

        return lf_input, hf_input, self.shallow(torch.cat([up_1, lf_input], 1)) # A2B: lf_feature, hf_feature, rc
        # return lf_input, hf_input, self.shallow(up_1+lf_input) # A2B: lf_feature, hf_feature, rc

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
    def __init__(self, A2B = True):
        super(shallowNet, self).__init__()
        if A2B:
          model = [nn.ReLU(True), nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64)]
        else:
          model = [nn.ReLU(True), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)]
        model += [ResnetBlock()]
        model += [nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
