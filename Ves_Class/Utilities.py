import numpy as np
import torch
import os
import random
import xlsxwriter
import pandas as pd
import pydicom as dcm
import torchvision.transforms as T
import cv2
import SimpleITK as sitk
from scipy.stats import norm
import sys
import copy
    
from scipy import ndimage

EPS = np.finfo(float).eps
    
import matplotlib.pyplot as plt

# import Loaders


def read_nii(file_name, current_index):
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)    
    file_reader.ReadImageInformation()
    sizeA=file_reader.GetSize()
    
    extract_size = (sizeA[0], sizeA[1], 1, 1)
        
    
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    
    img = sitk.GetArrayFromImage(file_reader.Execute())
    img = np.squeeze(img)
    
    # img = np.pad(img,((addX[2],addX[3]),(addX[0],addX[1]),(0,0)),'constant',constant_values=(-1024, -1024))


    return img


def size_nii(file_name):
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)    
    file_reader.ReadImageInformation()
    sizeA=file_reader.GetSize()
    
    return sizeA


def augmentation2(img, params):        
    angle = params[0]['Angle']
    translate = params[0]['Transl']
    scale = params[0]['Scale']
    shear = 0
    CenterCrop = params[0]['Crop_size']
    flip = params[0]['Flip']
    vel = params[0]['Output_size']
    
    if flip:
        img = torch.flip(img, [len(img.size())-1])
    
    # img = T.CenterCrop(size=CenterCrop)(img)
    augm_img = T.functional.affine(img, angle, translate, scale, shear,  T.InterpolationMode('nearest'))
    # augm_img = T.CenterCrop(size=CenterCrop)(augm_img)
    # resize = T.Resize((*scale,*scale), T.InterpolationMode('nearest'))
    # augm_img = resize(augm_img)
    # augm_img = resize_with_padding_Tensor(augm_img,(vel,vel))
    augm_img = T.CenterCrop(size=vel)(augm_img)


    return augm_img

def augmentation3(img, params, dc_pos):        
    angle = params[0]['Angle']
    translate = params[0]['Transl']
    scale = params[0]['Scale']
    shear = 0
    CenterCrop = params[0]['Crop_size']
    flip = params[0]['Flip']
    vel = params[0]['Output_size']
    augm_dc_pos = copy.deepcopy(dc_pos)
    
    if flip:
        img = torch.flip(img, [len(img.size())-1])
        augm_dc_pos[0] = img.shape[2] - augm_dc_pos[0]
    
    # img = T.CenterCrop(size=CenterCrop)(img)
    augm_img = T.functional.affine(img, angle, translate, scale, shear,  T.InterpolationMode('nearest'))
    # augm_img = T.CenterCrop(size=CenterCrop)(augm_img)
    # resize = T.Resize((*scale,*scale), T.InterpolationMode('nearest'))
    # augm_img = resize(augm_img)
    # augm_img = resize_with_padding_Tensor(augm_img,(vel,vel))
    # augm_img = T.CenterCrop(size=vel)(augm_img)
    
    R = np.array([[np.cos(np.deg2rad(angle)), - np.sin(np.deg2rad(angle)), translate[0]],
                  [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), translate[1]],
                  [0, 0, 1]])
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    Gx = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]])
    Gy = np.array([[1, 0, 0], [shear, 1, 0], [0, 0, 1]])
    A = Gx@Gy@S@R
    
    augm_dc_pos = A@np.array([[augm_dc_pos[0]], [augm_dc_pos[1]], [1]])
    
    augm_img_cpu = augm_img.detach().cpu().numpy().squeeze()
    augm_img_cpu = resize_with_padding_dc(augm_img_cpu, (vel,vel),
                                          [round(augm_dc_pos[0][0].tolist()),
                                           round(augm_dc_pos[1][0].tolist())])
    augm_img_cpu = np.expand_dims(augm_img_cpu, axis=0)
    augm_img = torch.from_numpy(augm_img_cpu)

    return augm_img

def crop_min(img):
    s = min( img.shape)
    img = img[0:s,0:s]
    return img


def crop_center(img, new_width=None, new_height=None):        
    width = img.shape[1]
    height = img.shape[0]
    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)  
        
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
        z = 1;
    else:
        center_cropped_img = img[top:bottom, left:right, ...]
        z = img.shape[2]   
        
    return center_cropped_img



def crop_center_final(img, new_width=None, new_height=None):        
    width = img.shape[1]
    height = img.shape[0]
    new_width_old = new_width
    new_heigh_old = new_height
    
    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)  
        
    new_width = min(width, new_width)
    new_height = min(height, new_height)
        
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
        z = 1;
    else:
        center_cropped_img = img[top:bottom, left:right, ...]
        z = img.shape[2] 
        
    padNUm=[] 
    padNUm.append(int(np.floor((new_width_old-center_cropped_img.shape[0])/2)))
    padNUm.append(int(np.ceil((new_width_old-center_cropped_img.shape[0])/2)))
    padNUm.append(int(np.floor((new_heigh_old-center_cropped_img.shape[1])/2)))
    padNUm.append(int(np.ceil((new_heigh_old-center_cropped_img.shape[1])/2)))
    padNUm = tuple(padNUm)
    
    center_cropped_img = np.pad(center_cropped_img, [padNUm[0:2],padNUm[2:4]], mode='constant', constant_values=(0, 0))
        
    return center_cropped_img, (top, bottom, left, right), padNUm


def Resampling(img, resO, resN, method='nearest'):   
    scF = (resO[0]/resN[0], resO[1]/resN[1])  
    velO = img.size()[1:3]
    velN = (int(velO[0]*scF[0]),int(velO[1]*scF[1]))
    
    resize = T.Resize((velN), T.InterpolationMode(method))
    img = resize(img)
    return img

def resize_with_padding(img, expected_size):
    delta_width = expected_size[0] - img.shape[0]
    delta_height = expected_size[1] - img.shape[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = np.array([pad_width, pad_height, delta_width - pad_width, delta_height - pad_height])
    padding[padding<0]=0
    img = np.pad(img, [(padding[0], padding[2]), (padding[1], padding[3])], mode='constant')
    img = crop_center(img, new_width=expected_size[0], new_height=expected_size[1])
    return img

def resize_with_padding_dc(img, expected_size, positions):
    delta_left = positions[0] - expected_size[0] // 2
    delta_right = positions[0] + expected_size[0] // 2
    delta_top = positions[1] - expected_size[1] // 2
    delta_bottom = positions[1] + expected_size[1] // 2
    
    if delta_left > 0:
        pos_left = delta_left
        delta_left = 0
    elif delta_left < 0:
        pos_left = 0
        delta_left = np.abs(delta_left)
    else:
        pos_left = 0
        delta_left = 0
        
    if delta_right > img.shape[1]:
        pos_right = delta_right + delta_left
        delta_right = delta_right - img.shape[1]
    elif delta_right == img.shape[1]:
        pos_right = delta_right + delta_left
        delta_right = 0
    else:
        pos_right = delta_right + delta_left
        delta_right = 0
        
    if delta_top > 0:
        pos_top = delta_top
        delta_top = 0
    elif delta_top < 0:
        pos_top = 0
        delta_top = np.abs(delta_top)
    else:
        pos_top = 0
        delta_top = 0
        
    if delta_bottom > img.shape[0]:
        pos_bottom = delta_bottom + delta_top
        delta_bottom = delta_bottom - img.shape[0]
    elif delta_bottom == img.shape[0]:
        pos_bottom = delta_bottom + delta_top
        delta_bottom = 0
    else:
        pos_bottom = delta_bottom + delta_top
        delta_bottom = 0    

    img_new = np.pad(img, ((delta_top, delta_bottom), (delta_left, delta_right)), mode='constant')
    img_new = img_new[pos_top:pos_bottom, pos_left:pos_right, ...]
    return img_new


def rot_transl(img, r, transl, flip=False, invers=False):
    
    velImg = np.size(img)
    
    img = torch.tensor(np.expand_dims(img, [0]).astype(np.float32))
    
    if invers:
        if flip:
            img = torch.flip(img, [len(img.size())-1])
        img = T.functional.affine(img, 0, (-transl[0],-transl[1]), 1.0, 0.0,  T.InterpolationMode('bilinear'))
        img = T.functional.affine(img, -r, (0,0), 1.0, 0.0,  T.InterpolationMode('bilinear'))
    else:
        img = T.functional.affine(img, r, (0,0), 1.0, 0.0,  T.InterpolationMode('bilinear'))
        img = T.functional.affine(img, 0, transl, 1.0, 0.0,  T.InterpolationMode('bilinear'))
        if flip:
            img = torch.flip(img, [len(img.size())-1])
    
    
    return img[0,:,:].detach().numpy()

def dice_loss(X, Y):
    eps = 0.00001
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return 1 - dice


def dice_coef(X, Y):
    # eps = 0.000001
    dice = ((2. * torch.sum(X*Y)) / (torch.sum(X) + torch.sum(Y)) )
    return dice

def dice_coef_batch(X, Y):
    eps = 0.000001
    dice = ((2. * torch.sum(X*Y,(1, 2)) + eps) / (torch.sum(X,(1, 2)) + torch.sum(Y,(1, 2)) + eps) )
    return dice

def acc_metric(X, Y):
    acc = np.sum(X == Y) / np.size(X)
    return acc

def MASD_compute(A, B):
    
    A  = A.astype(np.dtype('uint8'))
    A_ctr = A - cv2.dilate(A, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    distA = cv2.distanceTransform(255-A_ctr, cv2.DIST_L2, 3)
    
    B  = B.astype(np.dtype('uint8'))
    B_ctr = B - cv2.dilate(B, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    
    HD = np.mean(distA[B_ctr>0])  

    return HD

def rand_norm_distrb(N, mu, std, n_range):
    pd = norm(mu,std)
    rmin = pd.cdf(n_range[0])
    rmax =  pd.cdf(n_range[1])
    rUnif = (rmax-rmin)*np.random.rand(N) + rmin;
    fcn = pd.ppf(rUnif)
    return fcn



def save_to_excel(dataframe, root_dir, name):
    writer = pd.ExcelWriter(os.path.join(root_dir, '{}.xlsx'.format(name)),
    engine='xlsxwriter',
    datetime_format='yyyy-mm-dd',
    date_format='yyyy-mm-dd')
    sheet = name
    dataframe.to_excel(writer, sheet_name=sheet)
    
    worksheet = writer.sheets[sheet]
    worksheet.set_column('A:ZZ', 22)
    writer.save()
    
 

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 
   
    
   
def ToOneHot(ten, numClass=3 ):
    
    OH = torch.tensor(  np.zeros([  ten.shape[0] , numClass, ten.shape[2], ten.shape[3]])  )
    
    for ind in range(0,numClass):
        OH[:,ind,:,:] = torch.squeeze( ten==ind )
    
    return OH 