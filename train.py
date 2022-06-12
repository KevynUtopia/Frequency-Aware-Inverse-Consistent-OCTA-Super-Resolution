#!/usr/bin/python3
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
import warnings
warnings.filterwarnings('ignore')

import utils
from utils import set_requires_grad, weights_init_normal, ReplayBuffer, LambdaLR, save_sample, eval, eval_6m, train_eval
from model import UnetGenerator, FS_DiscriminatorA, FS_DiscriminatorB, phase_consistency_loss, NetworkA2B, NetworkB2A, PerceptualLoss, TVLoss
from dataset import ImageDataset, ImageDataset_6mm
import ssim


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default="./dataset/Colab_random_OCTA_augmented", help='root directory of the dataset')
parser.add_argument('--pretrained_root', type=str, default="./pre_trained/netG_A2B_pretrained.pth", help='root directory of the pre-trained model')
parser.add_argument('--pretrained', type=bool, default=False, help='whether use pre-trained model')
parser.add_argument('--B2A', type=bool, default=False, help='save netB2A')
parser.add_argument('--scheduler', type=bool, default=True, help='save netB2A')
parser.add_argument('--lr', type=float, default=1.3e-4, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--sizeA', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--sizeB', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

input_nc = opt.input_nc
output_nc = opt.output_nc
batchSize = opt.batchSize
size_A, size_B = opt.sizeA, opt.sizeB
lr = opt.lr
n_epochs, epoch, decay_epoch = opt.n_epochs, opt.epoch, opt.decay_epoch
n_cpu = opt.n_cpu
dataroot = opt.dataroot
cuda = True

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = NetworkA2B()
netG_B2A = NetworkB2A()
netD_A = FS_DiscriminatorA(input_nc)
netD_B = FS_DiscriminatorB(output_nc)

if cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

if not opt.pretrained:
    netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.SmoothL1Loss()
criterion_cycle = torch.nn.L1Loss()
criterion_phase = phase_consistency_loss()
criterion_identity = torch.nn.L1Loss()
criterion_perceptual = PerceptualLoss(torch.nn.MSELoss())
criterion_ssim = ssim.SSIM()
criterion_ssim_TV_loss= TVLoss().cuda()
criterion_feature = torch.nn.BCEWithLogitsLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.AdamW(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.9, 0.999))
optimizer_D = torch.optim.AdamW(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=lr, betas=(0.9, 0.999))

if opt.scheduler:
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step, verbose=True)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step, verbose=True)
else:
    lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=n_epochs, eta_min=0, verbose=True)
    lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=n_epochs, eta_min=0, verbose=True)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(batchSize, input_nc, size_A*2, size_A*2)
input_C = Tensor(batchSize, input_nc, size_A, size_A)
# input_B = Tensor(batchSize, output_nc, size_A, size_A)
input_B = Tensor(batchSize, output_nc, size_B, size_B)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

target_real = torch.flatten(target_real)
target_fake = torch.flatten(target_fake)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_A = [ 
                transforms.ToTensor(),
                transforms.RandomCrop((size_A, size_A)),
                transforms.Resize((size_A*2, size_A*2), interpolation=Image.BICUBIC),
                transforms.Normalize((0.5), (0.5))
                ]
                
transforms_B = [ 
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                transforms.RandomCrop((size_B, size_B))
                ]
dataset = ImageDataset(dataroot, transforms_A=transforms_A, transforms_B=transforms_B, unaligned=True)
print (len(dataset))


dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

###################################


test_path = "./dataset/evalution_6mm/parts"
transforms_A = [ 
                transforms.ToTensor(),
                # transforms.Normalize((0.246), (0.170)),
                 
                transforms.CenterCrop(256),
                transforms.Normalize((0.5), (0.5)),
                # transforms.Resize((128, 128))
                ]
transforms_B = [ 
                transforms.ToTensor(),
                # transforms.Normalize((0.246), (0.170)),
                transforms.Normalize((0.5), (0.5)),
                transforms.CenterCrop(256)]
test_dataset = ImageDataset_6mm(test_path, transforms_A=transforms_A, transforms_B=transforms_B, unaligned=True)

###### Training ######
for epoch in range(epoch, n_epochs):
    real_out, fake_out = None, None
    for i, batch in enumerate(dataloader):
        real_A = Variable(input_A.copy_(batch['A']))
        # real_C = Variable(input_C.copy_(batch['C']))
        real_B = Variable(input_B.copy_(batch['B']))

        ######### (1) forward #########

        ## G A->B##
        hf = utils.high_pass(real_A[0], i=10).unsqueeze(0).unsqueeze(0)
        hf = (hf+real_A)/2.0
        lf = utils.low_pass(real_A[0], i=7).unsqueeze(0).unsqueeze(0)
        lf_feature_A, hf_feature_A, fake_B = netG_A2B(lf, hf) # A2B: lf_feature, hf_feature, rc

        tv_fake_B = criterion_ssim_TV_loss(fake_B)*0.5
        ## idt A->A ##
        _, _, idt_A = netG_B2A(hf, lf)


        hf_feature_A = hf_feature_A.detach()
        hf_feature_A.requires_grad = False
        lf_feature_A = lf_feature_A.detach()
        lf_feature_A.requires_grad = False

        ## G B->A ##
        hf = utils.high_pass(fake_B[0], i=5).unsqueeze(0).unsqueeze(0)
        # TODO: low frequency boosting?
        hf = (hf+fake_B)/2.0
        lf = utils.low_pass(fake_B[0], i=12).unsqueeze(0).unsqueeze(0)
        # lf = (lf+fake_B)/2.0
        hf_feature_recovered_A, lf_feature_recovered_A, recovered_A = netG_B2A(hf, lf) # B2A: hf_feature, lf_feature, rc
        

        ## G B->A ##
        hf = utils.high_pass(real_B[0], i=5).unsqueeze(0).unsqueeze(0)
        hf = (hf+real_B)/2.0
        lf = utils.low_pass(real_B[0], i=12).unsqueeze(0).unsqueeze(0)
        # lf = (lf+real_B)/2.0
        hf_feature_B, lf_feature_B, fake_A = netG_B2A(hf, lf)

        ## idt B->B ##
        _, _, idt_B = netG_A2B(lf, hf)

        lf_feature_B = lf_feature_B.detach()
        lf_feature_B.requires_grad = False
        hf_feature_B = hf_feature_B.detach()
        hf_feature_B.requires_grad = False

        ## G A->B ##
        hf = utils.high_pass(fake_A[0], i=10).unsqueeze(0).unsqueeze(0)
        hf = (hf+fake_A)/2.0
        lf = utils.low_pass(fake_A[0], i=7).unsqueeze(0).unsqueeze(0)
        lf_feature_recovered_B, hf_feature_recovered_B, recovered_B = netG_A2B(lf, hf)


        ###### (2) G_A and G_B ######
        set_requires_grad([netD_A, netD_B], False)
        optimizer_G.zero_grad()

        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)




        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*15.0 + 2.0*criterion_feature(hf_feature_A, hf_feature_recovered_A) 
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*15.0 + 0.5*criterion_feature(hf_feature_B, hf_feature_recovered_B) 
        # TODO: verify identity loss, why it work? Mode collaps?
        loss_idt = criterion_identity(real_A, idt_A)*5.0 +  criterion_identity(real_B, idt_B)*5.0
        loss_perceptual = criterion_perceptual.get_loss(recovered_A.repeat(1,3,1,1), real_A.repeat(1,3,1,1))
        # loss_ssim = (1- criterion_ssim(recovered_A, real_A)) + (1 - criterion_ssim(recovered_B, real_B) )

        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_idt #+ loss_ssim

        loss_G.backward()        
        optimizer_G.step()

        ###### (3) D_A and D_B ######
        set_requires_grad([netD_A, netD_B], True)
        optimizer_D.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)
        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()


        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)      
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D.step()
        
        ####################################
        ####################################

        if i == 1:
            input_tmp = Tensor(batchSize, input_nc, size_A*2, size_A*2)
            x = Variable(input_tmp.copy_(batch['A']))
            real_out = x
            hf = utils.high_pass(x[0], i=10).unsqueeze(0).unsqueeze(0)
            hf = (hf+x)/2.0
            lf = utils.low_pass(x[0], i=8).unsqueeze(0).unsqueeze(0)
            _, _, fake_out = netG_A2B(lf, hf)
      
    save_sample(epoch, real_out, "_input")
    save_sample(epoch, fake_out, "_output")

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    
    if opt.pretrained:
        if (epoch<opt.decay_epoch and epoch%5==4) or (epoch>=opt.decay_epoch):
            torch.save(netG_A2B.state_dict(), './output_exp/netG_A2B_epoch'+str(epoch+1)+'.pth')
            if opt.B2A:
                torch.save(netG_B2A.state_dict(), './output_exp/netG_B2A_epoch'+str(epoch+1)+'.pth')
    else:
        if epoch%3==2:
            torch.save(netG_A2B.state_dict(), './output_exp/netG_A2B_epoch'+str(epoch+1)+'.pth')
            if opt.B2A:
                torch.save(netG_B2A.state_dict(), './output_exp/netG_B2A_epoch'+str(epoch+1)+'.pth')
            
    print("Epoch (%d/%d) Finished" % (epoch+1, n_epochs))
    eval(netG_A2B, epoch=epoch)
    eval_6m(netG_A2B, test_dataset)
    print("------------------------------------------")
