from torch.utils import data
import numpy as np
import torch 
import os
from skimage.io import imread
from glob import glob

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import laplace

import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray


from split_data import DataSpliter


def augmentation(img,mask,config):
    
    def rand(size=None):
        if size:
            return torch.rand(size).numpy()
        else:
        
            return torch.rand(1).numpy()[0]
    
    if config.deformation:
        if rand()>config.p:
            cols=img.shape[0]
            rows=img.shape[1]
            sr=config.scale_deform
            gr=config.shear_deform
            tr=0
            dr=100
            rr=180
            #sr = scales
            #gr = shears
            #tr = tilt
            #dr = translation
            sx=1+sr*rand()
            if rand()>0.5:
                sx=1/sx
            sy=1+sr*rand()
            if rand()>0.5:
                sy=1/sy
            gx=(0-gr)+gr*2*rand()
            gy=(0-gr)+gr*2*rand()
            tx=(0-tr)+tr*2*rand()
            ty=(0-tr)+tr*2*rand()
            dx=(0-dr)+dr*2*rand()
            dy=(0-dr)+dr*2*rand()
            t=(0-rr)+rr*2*rand()
            
            M=np.array([[sx, gx, dx], [gy, sy, dy],[tx, ty, 1]])
            R=cv2.getRotationMatrix2D((cols / 2, rows / 2), t, 1)
            R=np.concatenate((R,np.array([[0,0,1]])),axis=0)
            matrix= np.matmul(R,M)
        
            img = cv2.warpPerspective(img,matrix, (cols,rows),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            if len(img.shape)==2:
                img = np.expand_dims(img, 2)
            
            if not config.method == 'pretraining':
                mask = cv2.warpPerspective(mask,matrix, (cols,rows),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)
        
    # img = cv2.warpPerspective(img,matrix, (cols,rows),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-0.5)
    # if not config.method == 'pretraining':
    #     mask = cv2.warpPerspective(mask,matrix, (cols,rows),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    
    r=[torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(4,(1,1)).view(-1).numpy()]
    if r[0]:
        img=np.fliplr(img)
        if not config.method == 'pretraining':
            mask=np.fliplr(mask)
    if r[1]:
        img=np.flipud(img)
        if not config.method == 'pretraining':
            mask=np.flipud(mask) 
    img=np.rot90(img,k=r[2]) 
    if not config.method == 'pretraining':
        mask=np.rot90(mask,k=r[2])    
    
    if rand()>config.p:
        multipy=config.multipy
        multipy=1+rand()*multipy
        if rand()>0.5:
            img=img*multipy
        else:
            img=img/multipy
       
    if rand()>config.p:
        add=config.add     
        add=(1-2*rand())*add
        img=img+add
    
    
    if img.shape[2]>1:
        for slice_num in range(img.shape[2]):
            
            slice_ = img[:,:,slice_num]
            
            if rand()>config.p:
                multipy=0.1 
                multipy=1+rand()*multipy
                if rand()>0.5:
                    slice_=slice_*multipy
                else:
                    slice_=slice_/multipy
               
            if rand()>config.p:
                add=0.1     
                add=(1-2*rand())*add
                slice_=slice_+add
                
            img[:,:,slice_num] = slice_
    
    
    
    
    
    if rand()>0.5:
        bs_r=(-0.5,0.5)
        r=1-2*rand()
        if r<=0:
            par=bs_r[0]*r
            img=img-par*laplace(img)
        if r>0:
            par=bs_r[1]*r
            img=gaussian_filter(img,par)

    
    return img,mask



class Dataset(data.Dataset):


    def __init__(self, names,augment,crop,config,crop_same=False,data_type=None):
       
        self.names = names
        self.augment = augment
        self.crop = crop
        self.config = config
        self.crop_same = crop_same
        self.data_type = data_type
        
        self.names = self.names*config.multiply_dataset
        
        

    def __len__(self):
        return len(self.names)


    def __getitem__(self, idx):

        name = self.names[idx]
        
        if not self.config.method == 'pretraining':
            mask=imread(name)>0
            mask=mask.astype(np.float32)
            
        else:
            mask = None
            
        name_tmp = '_'.join(name.split('_')[:-1]) + '.png'
        name_tmp = name_tmp.replace('Vessels','Images').replace('Disc','Images').replace('Cup','Images')
            
        img=imread(name_tmp)
        img=img.astype(np.float64)
        img = img/255
        
        if self.config.img_type == 'rgb':
            pass
        elif self.config.img_type == 'green':
            img = img[:,:,1]
            img = np.expand_dims(img,2)
        elif self.config.img_type == 'gray':  
            img = rgb2gray(img)
            img = np.expand_dims(img,2)
        else:
            raise Exception('incorect image type')
        
        
        if self.augment:
            img,mask = augmentation(img,mask,self.config)
        
        
        in_size=img.shape
        out_size=[self.config.patch_size,self.config.patch_size]
        
        
        if self.crop:
            r1=torch.randint(in_size[0]-out_size[0],(1,1)).view(-1).numpy()[0]
            r2=torch.randint(in_size[1]-out_size[1],(1,1)).view(-1).numpy()[0]
            r=[r1,r2]
            
            if self.crop_same:
                r = [100,100]
            
            
            img=img[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1],:]
            if not self.config.method == 'pretraining':
                mask=mask[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]    
         
        
        if self.config.clahe:
            
            if img.shape[2]==1:
                
                img = img*255
                img[img<0] = 0
                img[img>255] = 255
                img=img.astype(np.uint8)
                
                clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip,tileGridSize=(self.config.clahe_grid,self.config.clahe_grid))
                img = clahe.apply(img[:,:,0])
                
                img = img.astype(np.float64)/255
                img = np.expand_dims(img,2)
                
            else:
                
                img = img*255
                img[img<0] = 0
                img[img>255] = 255
                img=img.astype(np.uint8)
                
                planes = cv2.split(img)
        
                clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip,tileGridSize=(self.config.clahe_grid,self.config.clahe_grid))
                
                planes[0] = clahe.apply(planes[0])
                planes[1] = clahe.apply(planes[1])
                planes[2] = clahe.apply(planes[2])
                
                img = cv2.merge(planes)
                
                img = img.astype(np.float64)/255
                
         
            img = img - 0.5
            
        if self.config.method =='pretraining':
            mask = img.copy()
            
            def rand():
                return torch.rand(1).numpy()[0]
            
            img_shape = img.shape
            
            if self.config.pretrain_noise_std_fraction:
                img = img + torch.randn(img_shape).numpy()*self.config.pretrain_std*self.config.pretrain_noise_std_fraction
                
            if self.config.pretrain_noise_pixel_p:
                
                tmp = torch.rand(img_shape).numpy()<self.config.pretrain_noise_pixel_p
                img[tmp] = 0
                img = img + (tmp).astype(np.float32) * torch.randn(img_shape).numpy() * self.config.pretrain_std*self.config.pretrain_noise_pixel_std_fraction
                
                
                
            block_types = ['chess' for _ in range(self.config.pretrain_chessboard_num_blocks)] + ['rot' for _ in range(self.config.pretrain_rot_num_blocks)] + ['del' for _ in range(self.config.pretrain_num_blocks)]
                
            p = torch.randperm(len(block_types)).numpy()
            
            block_types = [block_types[k] for k in p]
                
            used = np.zeros(img_shape[:2])
            for k,block_type in enumerate(block_types):
                
                if block_type == 'del':
                    block_sizex = int(np.ceil(rand()*self.config.pretrain_max_block_size))
                    block_sizey = int(np.ceil(rand()*self.config.pretrain_max_block_size))
                elif block_type == 'chess':
                    block_sizex = int(np.ceil(self.config.pretrain_chessboard_max_block_size/2 + rand()*self.config.pretrain_chessboard_max_block_size/2)/2)*2
                    # block_sizey = int(np.ceil(self.config.pretrain_chessboard_max_block_size/2 + rand()*self.config.pretrain_chessboard_max_block_size/2)/2)*2
                    block_sizey = block_sizex
                    
                else:
                    block_sizex = int(np.ceil(self.config.pretrain_chessboard_max_block_size/2 + rand()*self.config.pretrain_chessboard_max_block_size/2))
                    block_sizey = block_sizex
                
                posx = int(np.round(rand()*(img_shape[0]-block_sizex)))
                posy = int(np.round(rand()*(img_shape[1]-block_sizey)))
                
                if np.sum(used[posx:posx+block_sizex,posy:posy+block_sizey])==0:
                    used[posx:posx+block_sizex,posy:posy+block_sizey] = 1
                    
                    
                    if block_type == 'del':
                        std_tmp = self.config.pretrain_std
                        mean_tmp = self.config.pretrain_mean
                        
                        # std_tmp = np.std(img[posx:posx+block_sizex,posy:posy+block_sizey,:])
                        # mean_tmp = np.mean(img[posx:posx+block_sizex,posy:posy+block_sizey,:])
                        
                        block = torch.randn([block_sizex,block_sizey,img.shape[2]]).numpy()*std_tmp + mean_tmp
                    
                    if block_type == 'rot':
                        
                        block = img[posx:posx+block_sizex,posy:posy+block_sizey,:]
                        # if rand()>0.5:
                        #     block = block + 0.2
                        # else:
                        #     block = block - 0.2
                        
                        
                        r=[torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(4,(1,1)).view(-1).numpy()]
                        
                        if r[0]:
                            block=np.fliplr(block)
        
                        if r[1]:
                            block=np.flipud(block)
        
                        block=np.rot90(block,k=r[2]) 


                    if block_type == 'chess':
                        
                        block = img[posx:posx+block_sizex,posy:posy+block_sizey,:]
                        # if rand()>0.5:
                        #     block = block + 0.2
                        # else:
                        #     block = block - 0.2
                        
                        
                        p = torch.randperm(4).numpy()
                        
                        x_split = int(block_sizex/2)
                        y_split = int(block_sizey/2)
                        
                        sub_blocks = []

                        sub_blocks.append(block[:x_split,:y_split,:].copy())
                        sub_blocks.append(block[x_split:,:y_split,:].copy())
                        sub_blocks.append(block[:x_split,y_split:,:].copy())
                        sub_blocks.append(block[x_split:,y_split:,:].copy())
                        
                        
                        block[:x_split,:y_split,:] = sub_blocks[p[0]]
                        block[x_split:,:y_split,:] = sub_blocks[p[1]]
                        block[:x_split,y_split:,:] = sub_blocks[p[2]]
                        block[x_split:,y_split:,:] = sub_blocks[p[3]]
                    
                    
                    img[posx:posx+block_sizex,posy:posy+block_sizey,:] = block
                    
                    

        if len(mask.shape)==2:
            mask = mask.reshape([mask.shape[0],mask.shape[1],1])
        mask=torch.from_numpy(np.transpose(mask,(2,0,1)).astype(np.float32))
        img=torch.from_numpy(np.transpose(img,(2,0,1)).astype(np.float32))
        
        
        return img,mask







if __name__ == "__main__":
    
    
    from config import Config    
    
    config = Config()
    config.method = 'pretraining'
    
    config.pretrain_num_blocks = 20
    config.pretrain_max_block_size = 40
    config.pretrain_noise_std_fraction = 0.1
    config.pretrain_noise_pixel_p = 0.02
    config.pretrain_chessboard_num_blocks = 20
    config.pretrain_chessboard_max_block_size = 40
    config.pretrain_rot_num_blocks = 20
    config.pretrain_rot_max_block_size = 40
    
    
    
    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=42)
    
    
    train_generator = Dataset(data_split['train'],augment=True,crop=True,config=config)
    train_generator = data.DataLoader(train_generator,batch_size=1,num_workers= 0, shuffle=True,drop_last=True)
    

    for img,mask in train_generator:
        
        plt.imshow(np.transpose(img[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        plt.show()
        plt.imshow(np.transpose(mask[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        plt.show()
        
        break
    
    
    
    
    
    from config import Config    
    
    config = Config()
    config.method = 'segmentation'
    
    config.pretrain_num_blocks = 20
    config.pretrain_max_block_size = 40
    config.pretrain_noise_std_fraction = 0.1
    config.pretrain_noise_pixel_p = 0.02
    config.pretrain_chessboard_num_blocks = 20
    config.pretrain_chessboard_max_block_size = 40
    config.pretrain_rot_num_blocks = 20
    config.pretrain_rot_max_block_size = 40
    
    
    
    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=42)
    
    
    train_generator = Dataset(data_split['train'],augment=True,crop=True,config=config)
    train_generator = data.DataLoader(train_generator,batch_size=1,num_workers= 0, shuffle=True,drop_last=True)
    

    for img,mask in train_generator:
        
        plt.imshow(np.transpose(img[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        plt.show()
        plt.imshow(np.transpose(mask[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        plt.show()
        
        break
    