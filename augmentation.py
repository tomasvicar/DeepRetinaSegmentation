import torch
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import laplace


def augmentation(img,mask,config):
    
    def rand(size=None):
        if size:
            return torch.rand(size).numpy()
        else:
        
            return torch.rand(1).numpy()[0]
    
    
    
    r = [torch.randint(2,(1,1)).view(-1).numpy(), torch.randint(2,(1,1)).view(-1).numpy(), torch.randint(4,(1,1)).view(-1).numpy()]
    if r[0]:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if r[1]:
        img = np.flipud(img)
        mask = np.flipud(mask) 
    img = np.rot90(img,k=r[2]) 
    mask = np.rot90(mask,k=r[2])    
        
    
    
    if config.deformation:
        if rand() > config.p:
            cols = img.shape[0]
            rows = img.shape[1]
            sr = config.scale_deform
            gr = config.shear_deform
            tr = 0
            dr = 0
            
            if config.rotate:
                rr = 180
            else:
                rr = 0
            #sr = scales
            #gr = shears
            #tr = tilt
            #dr = translation
            sx = 1 + sr * rand()
            if rand() > 0.5:
                sx = 1 / sx
            sy = 1 + sr * rand()
            if rand() > 0.5:
                sy = 1 / sy
            gx = (0 - gr) + gr * 2 * rand()
            gy = (0 - gr) + gr * 2 *rand()
            tx = (0 - tr) + tr * 2 * rand()
            ty = (0 - tr) + tr * 2 * rand()
            dx = (0 - dr) + dr * 2 * rand()
            dy = (0 - dr) + dr * 2 * rand()
            r = (0 - rr) + rr * 2 * rand()
            
            M = np.array([[sx, gx, dx], [gy, sy, dy],[tx, ty, 1]])
            R = cv2.getRotationMatrix2D((cols / 2, rows / 2), r, 1)
            R = np.concatenate((R, np.array([[0,0,1]])),axis=0)
            matrix = np.matmul(R, M)
        
            img = cv2.warpPerspective(img, matrix, (cols,rows),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpPerspective(mask, matrix, (cols,rows),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)
    
    
    
    in_size=img.shape
    out_size=[config.patch_size,config.patch_size]
    r1=int((in_size[0]-out_size[0])/2)
    r2=int((in_size[1]-out_size[1])/2)
    r=[r1,r2]
    img=img[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1],:]
    mask=mask[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
    
    
    if rand() > config.p:
        multipy = config.multipy
        multipy = 1 + rand()*multipy
        if rand() > 0.5:
            img = img * multipy
        else:
            img = img / multipy
       
    if rand() > config.p:
        add = config.add     
        add = (1 - 2*rand()) * add
        img = img + add
    
    if rand() > config.p:
        if img.shape[2] > 1:
            for slice_num in range(img.shape[2]):
                
                slice_ = img[:,:,slice_num]
                multipy = 0.1 
                multipy = 1 + rand() * multipy
                if rand() > 0.5:
                    slice_ = slice_ * multipy
                else:
                    slice_ = slice_ / multipy
                
       
    if rand() > config.p:
        if img.shape[2] > 1:
            for slice_num in range(img.shape[2]):
                
                slice_ = img[:,:,slice_num]
                add = 0.1     
                add = (1 - 2 * rand()) * add
                slice_ = slice_ + add
                img[:,:,slice_num] = slice_
            
    if rand()>config.p:
        bs_r =(-config.sharp,config.blur)
        r = 1 - 2 * rand()
        if r <= 0:
            par = bs_r[0] * r
            img = img - par * laplace(img)
        if r > 0:
            par = bs_r[1] * r
            img = gaussian_filter(img, par)
    

    return img.copy(), mask.copy() 