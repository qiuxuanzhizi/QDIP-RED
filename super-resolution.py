#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from models import *
import torch
import torch.optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models.downsampler import Downsampler
from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1 
factor = 4 # 8
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True
path_to_image =  "C:/Users/Desktop/image0104/1.png"
# # Load image and baselines

# Starts here
imgs = load_LR_HR_imgs_sr(path_to_image , imsize, factor, enforse_div32)
imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

if PLOT:
    plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12);
    print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                                        compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
                                        compare_psnr(imgs['HR_np'], imgs['nearest_np'])))


# # Convert real to quaternion
grayscale = torchvision.transforms.Grayscale(num_output_channels=1)

def convert_data_for_quaternion(img):
    """
    converts batches of RGB images in 4 channels for QNNs
    """
    #img_quaternion=torch.cat([torch.zeros([1,1,img.shape[2],img.shape[3]]).to(device),img] ,1)
    img_quaternion=torch.cat([img,grayscale(img)] ,1)
    return img_quaternion
img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)
img_LR_var = convert_data_for_quaternion(img_LR_var)
img_HR_var = np_to_torch(imgs['HR_np']).type(dtype)
img_HR_var = convert_data_for_quaternion(img_HR_var)

print(img_LR_var.shape)
print(img_HR_var.shape)


# # Set up parameters and net
input_depth = 32
 
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'

LR = 0.01
tv_weight = 0.9

OPTIMIZER = 'adam'

if factor == 4: 
    num_iter = 1000
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'
    
net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

NET_TYPE = 'qskip'
net = qskip(
                input_depth, 4, 
                num_channels_down = [128,128,128,128,128], 
                num_channels_up   = [128,128,128,128,128],
                num_channels_skip = [4, 4, 4, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)


# Losses
mse = torch.nn.MSELoss().type(dtype)

#img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)

downsampler = Downsampler(n_planes=4, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

save_path='D:/python_learning/QDIP/result/denoise/QDIP_bee.png'
import matplotlib
out_avg = None
exp_weight=0.99
def closure():
    global i, net_input,out_avg
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_HR = net(net_input)
    out_LR = downsampler(out_HR[0:3,:,:])
    
    if out_avg is None:
        out_avg = out_HR.detach()
    else:
        out_avg = out_avg * exp_weight + out_HR.detach() * (1 - exp_weight)

    total_loss = mse(out_LR, img_LR_var[0:3,:,:]) 
    
  #  if tv_weight > 0:
  #      total_loss += tv_weight * tv_loss(out_HR)
        
    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR)[0:3,:,:])
    psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR)[0:3,:,:])
    psnr_avg = compare_psnr(imgs['HR_np'][0:3,:,:], out_avg.detach().cpu().numpy()[0][0:3,:,:])
    print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f PSNR_Avg %.3f' % (i, psnr_LR, psnr_HR, psnr_avg), '\r', end='')
                      
    # History
    psnr_history.append([psnr_LR, psnr_HR])
    
    if PLOT and i % 100 == 0:
        out_HR_np = torch_to_np(out_HR)
        out_avg_np=np.clip(torch_to_np(out_avg)[0:3,:,:],0,1)
        plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np[0:3,:,:], 0, 1)], factor=13, nrow=3)
        out_avg_np1 = out_avg_np.transpose(1,2,0)
        matplotlib.image.imsave(save_path,out_avg_np1)
        #matplotlib.image.imsave(save_path1,imgs['LR_np'])

    i += 1
    
    return total_loss

psnr_history = [] 
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
psnr_history = [] 
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
save_path1='D:/python_learning/QDIP/result/denoise/noi_bee.png'
matplotlib.image.imsave(save_path1,imgs['LR_np'].transpose(1,2,0))
imgs['LR_np'].shape