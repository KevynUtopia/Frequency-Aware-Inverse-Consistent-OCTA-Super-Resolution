import argparse
import glob
import random
import os
from PIL import Image
import numpy as np
import time
import datetime
import sys
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import itertools
import matplotlib.pyplot as plt
import pdb
import skimage.metrics
from tqdm import tqdm
import cv2

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

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
    return iimg*-1

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

def laplacian_kernel(im):
    conv_op = nn.Conv2d(1, 1, 3, bias=False, stride=1, padding=1)
    laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    laplacian_kernel = laplacian_kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = torch.from_numpy(laplacian_kernel).cuda()
    edge_detect = conv_op(Variable(im))
    return edge_detect

def functional_conv2d(im):
    sobel_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge_detect = F.conv2d(Variable(im), weight)
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save_sample(epoch, tensor, suffix="_real"):
    output = tensor.cpu().detach().numpy().squeeze(0).squeeze(0)
    plt.imsave('./checkpoint_exp/image_alt_'+str(epoch+1)+suffix+'.jpeg', output, cmap="gray")

def eval(model):
    lr = "./dataset/test/6x6_256/"
    hr = "./dataset/test/3x3_256/"
    num, psnr, ssim, mse, nmi= 0, 0, 0, 0, 0
    T_1 = transforms.Compose([ transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                transforms.Resize([128, 128])
                 ])
    T_2 = transforms.Compose([ transforms.ToTensor(),                         
                  transforms.Normalize((0.5), (0.5))])
    for i in tqdm(range(297)):
        lr_path = os.path.join(lr, str(i)+"_3.png")
        hr_path = os.path.join(hr, str(i)+"_6.png")
        if os.path.isfile(lr_path) and os.path.isfile(hr_path):
            lr_img = Image.open(lr_path).convert('L')
            hr_img = Image.open(hr_path).convert('L')
            
            lr_img = T_1(lr_img).cuda().unsqueeze(0)
            hr_img = T_2(hr_img).cuda().unsqueeze(0)
            
            _, _, sr_img = model(lr_img)

            yimg = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            gtimg = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
            ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
            mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
            nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
            num += 1
    print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f"%(psnr/num, ssim/num, mse/num, nmi/num))


def eval_6m(model, dataset):
    n = len(dataset)
    num, psnr, ssim, mse, nmi= 0, 0, 0, 0, 0
    for i in range(n):
        img = dataset[i]['A'].unsqueeze(0).cuda()
        gt = dataset[i]['B'].unsqueeze(0).cuda()


        _, _, y = model(img)

        yimg = y.cpu().detach().numpy().squeeze(0).squeeze(0)
        
        # print(gt.shape)
        # print("fdsafasd fsdaf")
        gtimg = gt.cpu().detach().numpy().squeeze(0).squeeze(0)
        psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
        ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
        mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
        nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
        num += 1
    print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f"%(psnr/num, ssim/num, mse/num, nmi/num))

def eval_6m_baseline(model, dataset):
    n = len(dataset)
    num, psnr, ssim, mse, nmi= 0, 0, 0, 0, 0
    for i in range(n):
        img = dataset[i]['A'].unsqueeze(0).cuda()
        gt = dataset[i]['B'].unsqueeze(0).cuda()


        y = model(img)

        yimg = y.cpu().detach().numpy().squeeze(0).squeeze(0)
        
        # print(gt.shape)
        # print("fdsafasd fsdaf")
        gtimg = gt.cpu().detach().numpy().squeeze(0).squeeze(0)
        psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
        ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
        mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
        nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
        num += 1
    print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f"%(psnr/num, ssim/num, mse/num, nmi/num))
    
def save_sample(epoch, tensor, suffix="_real"):
    output = tensor.cpu().detach().numpy().squeeze(0).squeeze(0)
    plt.imsave('./checkpoint_exp/image_alt_'+str(epoch+1)+suffix+'.jpeg', output, cmap="gray")