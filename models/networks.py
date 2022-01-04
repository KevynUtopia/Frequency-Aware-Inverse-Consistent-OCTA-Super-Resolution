import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import glob
import random
import os
from PIL import Image
import numpy as np
import random
import time
import datetime
import sys



###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def guais_low_pass(img, radius=10):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask[i, j] = np.exp(-0.5 *  distance_u_v / (radius ** 2))
    return torch.from_numpy(mask).float()


def guais_high_pass(img, radius=10):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask[i, j] = 1 - np.exp(-0.5 *  distance_u_v / (radius ** 2))
    return torch.from_numpy(mask).float()


def high_pass(timg, i=4):
    f = torch.fft.fft2(timg[0])
    fshift = torch.fft.fftshift(f)
    
    # rows, cols = timg[0].shape
    # crow, ccol = int(rows/2), int(cols/2)
    # fshift[crow-i:crow+i, ccol-i:ccol+i] = 0
    mask = guais_high_pass(fshift, i).cuda()
    f = fshift * mask

    ishift = torch.fft.ifftshift(f)
    iimg = torch.fft.ifft2(ishift)
    iimg = torch.abs(iimg)
    return iimg

def low_pass(timg, i=10):
    f = torch.fft.fft2(timg[0])
    fshift = torch.fft.fftshift(f)
    
    # print(timg[0].shape)
    rows, cols = timg[0].shape
    
    crow,ccol = int(rows/2), int(cols/2)
    # mask = torch.zeros((rows, cols)).cuda()
    # mask[crow-i:crow+i, ccol-i:ccol+i] = 1
    mask = guais_low_pass(fshift, i).cuda()
    
    
    f = fshift * mask
    
    ishift = torch.fft.ifftshift(f)
    iimg = torch.fft.ifft2(ishift)
    iimg = torch.abs(iimg)
    return iimg

def bandreject_pass(timg, r_out=300, r_in=35):
    f = torch.fft.fft2(timg[0])
    fshift = torch.fft.fftshift(f)
    
    rows, cols = timg[0].shape
    crow,ccol = int(rows/2), int(cols/2)
    # mask = torch.zeros((rows, cols)).cuda()
    # mask[crow-i:crow+i, ccol-i:ccol+i] = 1
    mask = bandreject_filters(fshift, r_out, r_in).cuda()
    
    f = fshift * mask
    
    ishift = torch.fft.ifftshift(f)
    iimg = torch.fft.ifft2(ishift)
    iimg = torch.abs(iimg)
    return iimg

def bandreject_filters(img, r_out=300, r_in=35):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    radius_out = r_out
    radius_in = r_in

    mask = np.zeros((rows, cols))
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                               ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1
    return torch.from_numpy(mask).float()

class phase_consistency_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        # print(x.size())
        radius = 15
        rows, cols = x[0][0].shape
        center = int(rows / 2), int(cols / 2)

        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
                mask[i, j] = np.exp(-0.5 *  distance_u_v / (radius ** 2))
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

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], A2B = True):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    
    if A2B:
        net = UnetGeneratorA2B(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        net = UnetGeneratorB2A(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class UnetGeneratorA2B(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGeneratorA2B, self).__init__()

        ####################### UNet #######################
        ######################################################
        # # (1, 320) -> (64, 160)
        # self.down_1 = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True)])
        # (64, 160) -> (128, 80)
        self.down_1 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(128)])
        # (128, 80) -> (256, 40)
        self.down_2 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(256)])
        # (256, 40) -> (512, 20)
        self.down_3 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(512)])
        # (512, 20) -> (1024, 10)
        self.down_4 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(1024)])

        # (1024, 10) -> (1024, 5)
        self.down_5 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=True)])
        
        ######################################################
        # (1024, 5) -> (1024, 10)
        self.up_5 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(1024)])
        # (1024, 10)*2 -> (512, 20)
        self.up_4 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(512)])
        # (512, 20)*2 -> (256, 40)
        self.up_3 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(256)])
        
        # (256, 40)*2 -> (128, 80)
        self.up_2 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(128)])
        # (128, 80)*2 -> (64, 160)
        self.up_1 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(64)])

        ######################################################
        ######################################################

        self.shallow_frequency = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.BatchNorm2d(64)])
        
        self.shallow = nn.Sequential(nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.Tanh()]))

    def forward(self, input):
        """Standard forward"""
        hf = high_pass(input[0], i=10).unsqueeze(0).unsqueeze(0) # (1, 320)
        lf = low_pass(input[0], i=8).unsqueeze(0).unsqueeze(0) # (1, 320)

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

        lf_input = self.shallow_frequency(lf)
        return down_1, self.shallow(up_1+lf_input)


class UnetGeneratorB2A(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGeneratorB2A, self).__init__()

        ####################### UNet #######################
        ######################################################
        # (1, 320) -> (64, 160)
        self.down_1 = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True)])
        # (64, 160) -> (128, 80)
        self.down_2 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(128)])
        # (128, 80) -> (256, 40)
        self.down_3 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(256)])
        # (256, 40) -> (512, 20)
        self.down_4 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(512)])
        # (512, 20) -> (1024, 10)
        self.down_5 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(1024)])

        # (1024, 10) -> (1024, 5)
        self.down_6 = nn.Sequential(*[nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=True)])
        
        ######################################################
        # (1024, 5) -> (1024, 10)
        self.up_6 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(1024)])
        # (1024, 10)*2 -> (512, 20)
        self.up_5 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(512)])
        # (512, 20)*2 -> (256, 40)
        self.up_4 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(256)])
        
        # (256, 40)*2 -> (128, 80)
        self.up_3 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(128)])
        # (128, 80)*2 -> (64, 160)
        self.up_2 = nn.Sequential(*[nn.ReLU(True), 
                                    nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(64)])

        ######################################################
        ######################################################
        self.shallow_frequency = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(64)])
        
        self.shallow = nn.Sequential(nn.Sequential(*[nn.ReLU(True), 
                                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.Tanh()]))

    def forward(self, input):
        """Standard forward"""
        hf = high_pass(input[0]).unsqueeze(0).unsqueeze(0) # (1, 320)
        lf = low_pass(input[0]).unsqueeze(0).unsqueeze(0) # (1, 320)

        down_1 = self.down_1(hf) # (64, 160)
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

        lf_input = self.shallow_frequency(lf) # (64, 160)
        return up_3, self.shallow(up_2+lf_input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
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
            model = down + [submodule] + up
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
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

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


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
