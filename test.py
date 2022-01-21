import torch
from model import UnetGeneratorA2B

netG_A2B = torch.load('./output_exp/netG_A2B_epoch60.pth')
# type(netG_A2B)
model = UnetGeneratorA2B(opt.output_nc, opt.input_nc).cuda()
model.load_state_dict(netG_A2B, strict=False)