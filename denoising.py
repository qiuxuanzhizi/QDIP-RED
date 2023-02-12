#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function  # 把python新版本中print_function函数的特性导入到当前版本
import matplotlib.pyplot as plt 
#%matplotlib inline  # 将matplotlib的图表直接嵌入到notebook中
from htorch import quaternion, layers, utils
import torchvision
import matplotlib
import xlwt
import os
import numpy as np
from models import *
import torch
import torch.optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.

fname = "C:/Users/Desktop/image0104/1.png"

img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)
    
img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
if PLOT:
    plot_image_grid([img_np[0:3,:,:], img_noisy_np[0:3,:,:]], 4, 6);


# # Convert the RGB image into quaternion domain 
grayscale = torchvision.transforms.Grayscale(num_output_channels=1)

def convert_data_for_quaternion(img):
    """
    converts batches of RGB images in 4 channels for QNNs
    """
    #img_quaternion=torch.cat([torch.zeros([1,1,img.shape[2],img.shape[3]]).to(device),img] ,1)
    img_quaternion=torch.cat([img,grayscale(img)] ,1)
    return img_quaternion


img_np_1 = np.concatenate((img_np,pil_to_np(grayscale(img_pil))),axis=0)
img_noisy_np_1 = np.concatenate((img_noisy_np,pil_to_np(grayscale(img_noisy_pil))),axis=0)


INPUT = 'noise'
pad = 'reflection'
OPT_OVER = 'net'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

reg_noise_std = 1./30 # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
exp_weight=0.99


num_iter = 5000
input_depth = 32 
figsize = 4 
    
net = qskip(
                input_depth, 4, 
                num_channels_down = [128,128,128,128,128], 
                num_channels_up   = [128,128,128,128,128],
                num_channels_skip = [4, 4, 4, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
#net_input = convert_data_for_quaternion(net_input) 


# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)
img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
img_noisy_torch = convert_data_for_quaternion(img_noisy_torch)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0
save_path='D:/python_learning/QDIP/result/denoise/QDIP_bird_25'
loss_list = []
psrn_list = []

i = 0
def closure():
    
    global i, out_avg, psrn_noisy_last, last_net, net_input
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()
    total_loss_np = total_loss.detach().cpu().numpy().ravel()
    loss_list.append(total_loss_np)
    
    psrn_noisy = compare_psnr(img_noisy_np_1[0:3,:,:], out.detach().cpu().numpy()[0][0:3,:,:]) 
    psrn_gt    = compare_psnr(img_np_1[0:3,:,:], out.detach().cpu().numpy()[0][0:3,:,:]) 
    psrn_gt_sm = compare_psnr(img_np_1[0:3,:,:], out_avg.detach().cpu().numpy()[0][0:3,:,:])
    psrn_list.append(psrn_gt_sm)
    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        out_avg_np=np.clip(torch_to_np(out_avg)[0:3,:,:],0,1)
        plot_image_grid([np.clip(out_np[0:3,:,:], 0, 1), 
                         np.clip(torch_to_np(out_avg)[0:3,:,:], 0, 1)], factor=figsize, nrow=1)
        
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

    return total_loss,loss_list

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

file=xlwt.Workbook('endcoding=utf-8')
sheet1=file.add_sheet('sheet1',cell_overwrite_ok=True)
sheet1.write(0,0,'num_iter')
sheet1.write(0,1,'psrn')
for i in range(len(loss_list)):
    sheet1.write(i+1,0,i)
    sheet1.write(i+1,1,psrn_list[i])
    
file.save('Data_psrn.xls')

file=xlwt.Workbook('endcoding=utf-8')
sheet1=file.add_sheet('sheet1',cell_overwrite_ok=True)
sheet1.write(0,0,'num_iter')
sheet1.write(0,1,'psrn')
for i in range(len(loss_list)):
    sheet1.write(i+1,0,i)
    sheet1.write(i+1,1,loss_list[i][0].astype(np.float64))
    
file.save('Data_loss.xls')