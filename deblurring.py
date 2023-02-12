#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function  # 把python新版本中print_function函数的特性导入到当前版本
import matplotlib.pyplot as plt 
#%matplotlib inline  # 将matplotlib的图表直接嵌入到notebook中
from htorch import quaternion, layers, utils
import torchvision
import os
import numpy as np
from models import *
import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.utils import *  # auxiliary functions
from utils.blur_utils import *  # blur functions
from utils.data import Data  # class that holds img, psnr, time
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True



NOISE_SIGMA = 2**.5  # sqrt(2), I haven't tests other options
BLUR_TYPE = 'uniform_blur'  # 'gauss_blur' or 'uniform_blur' that the two only options
GRAY_SCALE = False  # if gray scale is False means we have rgb image, the psnr will be compared on Y. ch.
                    # if gray scale is True it will turn rgb to gray scale
USE_FOURIER = False

# graphs labels:
X_LABELS = ['Iterations']*3
Y_LABELS = ['PSNR between x and net (db)', 'PSNR with original image (db)', 'loss']

# Algorithm NAMES (to get the relevant image: use data_dict[alg_name].img)
# for example use data_dict['Clean'].img to get the clean image
ORIGINAL  = 'Clean'
CORRUPTED = 'Blurred'

def load_imgs_deblurring(fname, blur_type, noise_sigma, plot=False):
    """  Loads an image, and add gaussian blur
    Args: 
         fname: path to the image
         blur_type: 'uniform' or 'gauss'
         noise_sigma: noise added after blur
         covert2gray: should we convert to gray scale image?
         plot: will plot the images
    Out:
         dictionary of images and dictionary of psnrs
    """
    img_pil, img_np = load_and_crop_image(fname)        # load
    if GRAY_SCALE:
        img_np = rgb2gray(img_pil)
    blurred = np.clip(blur(img_np, blur_type),0,1)  # blur, and the line below adds noise
    data_dict = { ORIGINAL: Data(img_np), 
                 CORRUPTED: Data(blurred, compare_PSNR(img_np, blurred, on_y=(not GRAY_SCALE), gray_scale=GRAY_SCALE))}
    if plot:
        plot_dict(data_dict)
    return data_dict

# Get the LR and HR images
data_dict = load_imgs_deblurring("C:/Users/Desktop/demof/House.png", BLUR_TYPE, NOISE_SIGMA, plot=True)

print(data_dict[CORRUPTED].img.shape)
print(data_dict[ORIGINAL].img.shape)

grayscale = torchvision.transforms.Grayscale(num_output_channels=1)

def convert_data_for_quaternion(img):
    """
    converts batches of RGB images in 4 channels for QNNs
    """
    #img_quaternion=torch.cat([torch.zeros([1,1,img.shape[2],img.shape[3]]).to(device),img] ,1)
    img_quaternion=torch.cat([img,grayscale(img)] ,1)
    return img_quaternion

img_original_torch = np_to_torch(data_dict[ORIGINAL].img).type(dtype)
img_corrupted_torch = np_to_torch(data_dict[CORRUPTED].img).type(dtype)
img_original_torch = convert_data_for_quaternion(img_original_torch).type(dtype)
img_corrupted_torch = convert_data_for_quaternion(img_corrupted_torch).type(dtype)
print(img_original_torch.size())

INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 0.004

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
exp_weight=0.99


num_iter = 15000
input_depth = 32 
figsize = 4 
    
net = qskip(
                input_depth, 4, 
                num_channels_down = [128,128,128,128,128], 
                num_channels_up   = [128,128,128,128,128],
                num_channels_skip = [4, 4, 4, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    
net_input = get_noise(input_depth, INPUT, (img_corrupted_torch.size()[2], img_corrupted_torch.size()[3])).type(dtype).detach()
#net_input = convert_data_for_quaternion(net_input) 


# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

H = get_h(4, BLUR_TYPE, USE_FOURIER, dtype)


import matplotlib
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
save_path = 'D:/python_learning/DIP/deep-image-prior-master/result/denoise/House.png'
out_avg = None
last_net = None
psrn_noisy_last = 0
noise_factor = 0.01


i = 0
def closure():
    
    global i, out_avg, psrn_noisy_last, last_net, net_input
    
    net_input = net_input_saved + (noise.normal_() * noise_factor)
    out = net(net_input)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
    total_loss = mse(H(out),img_corrupted_torch)
    total_loss.backward()
        
    
    psrn_noisy = compare_psnr(data_dict[CORRUPTED].img[0:3,:,:], out.detach().cpu().numpy()[0][0:3,:,:]) 
    psrn_gt    = compare_psnr(data_dict[ORIGINAL].img[0:3,:,:], out.detach().cpu().numpy()[0][0:3,:,:]) 
    psrn_gt_sm = compare_psnr(data_dict[ORIGINAL].img[0:3,:,:], out_avg.detach().cpu().numpy()[0][0:3,:,:]) 
    
    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np[0:3,:,:], 0, 1), 
                         np.clip(torch_to_np(out_avg)[0:3,:,:], 0, 1)], factor=figsize, nrow=1)
        out_avg_np = np.clip(torch_to_np(out_avg)[0:3,:,:],0,1)
        out_avg_np1 = out_avg_np.transpose(1,2,0)
        matplotlib.image.imsave(save_path,out_avg_np1)
        
    
    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy
            
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


