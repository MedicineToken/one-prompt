
import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from conf import settings
import time
import cfg
from conf import settings
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch
from einops import rearrange
import pytorch_ssim

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)


import torch


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_one(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:

        for pack in train_loader:
            if ind == 0:
                tmp_img = pack['image'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
                tmp_mask = pack['label'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
                if 'pt' not in pack:
                    tmp_img, pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']
                
                if point_labels[0] != -1:
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)

            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)

            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            with torch.no_grad():
                imge, skips= net.image_encoder(imgs)
                timge, tskips = net.image_encoder(tmp_img)

                    # imge= net.image_encoder(imgs)
            p1, p2, se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    doodles= None,
                    masks=None,
                )

            pred, _ = net.mask_decoder(
                skips_raw = skips,
                skips_tmp = tskips,
                raw_emb = imge,
                tmp_emb = timge,
                pt1 = p1,
                pt2 = p2,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
              )

            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False)

            pbar.update()

    return loss

def validation_one(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        
        for ind, pack in enumerate(val_loader):
            if ind == 0:
                tmp_img = pack['image'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
                tmp_mask = pack['label'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
                if 'pt' not in pack:
                    tmp_img, pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']
                
                if point_labels[0] != -1:
                    # point_coords = onetrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)


            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)

            name = pack['image_meta_dict']['filename_or_obj']
            
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            '''test'''
            with torch.no_grad():
                imge, skips= net.image_encoder(imgs)
                timge, tskips = net.image_encoder(tmp_img)

                p1, p2, se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        doodles= None,
                        masks=None,
                    )
                pred, _ = net.mask_decoder(
                    skips_raw = skips,
                    skips_tmp = tskips,
                    raw_emb = imge,
                    tmp_emb = timge,
                    pt1 = p1,
                    pt2 = p2,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
            
                tot += lossfunc(pred, masks)

                '''vis images'''
                if ind % args.vis == 0:
                    namecat = 'Test'
                    for na in name:
                        img_name = na.split('/')[-1].split('.')[0]
                        namecat = namecat + img_name + '+'
                    vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False)
                

                temp = eval_seg(pred, masks, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()


    return tot/ n_val , tuple([a/n_val for a in mix_res])
