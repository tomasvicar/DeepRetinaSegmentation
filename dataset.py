from torch.utils import data
import numpy as np
import torch 
import time
import h5py
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import laplace


from split_data import DataSpliter


def augmentation(img,mask,config):
    
    def rand(size=None):
        if size:
            return torch.rand(size).numpy()
        else:
        
            return torch.rand(1).numpy()[0]
    
    
    
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
        
        
    # if config.deformation:
    #     if rand()>config.p:
    #         cols=img.shape[0]
    #         rows=img.shape[1]
    #         sr=config.scale_deform
    #         gr=config.shear_deform
    #         tr=0
    #         dr=10
            
    #         if config.rotate:
    #             rr=180
    #         else:
    #             rr = 0
    #         #sr = scales
    #         #gr = shears
    #         #tr = tilt
    #         #dr = translation
    #         sx=1+sr*rand()
    #         if rand()>0.5:
    #             sx=1/sx
    #         sy=1+sr*rand()
    #         if rand()>0.5:
    #             sy=1/sy
    #         gx=(0-gr)+gr*2*rand()
    #         gy=(0-gr)+gr*2*rand()
    #         tx=(0-tr)+tr*2*rand()
    #         ty=(0-tr)+tr*2*rand()
    #         dx=(0-dr)+dr*2*rand()
    #         dy=(0-dr)+dr*2*rand()
    #         t=(0-rr)+rr*2*rand()
            
    #         M=np.array([[sx, gx, dx], [gy, sy, dy],[tx, ty, 1]])
    #         R=cv2.getRotationMatrix2D((cols / 2, rows / 2), t, 1)
    #         R=np.concatenate((R,np.array([[0,0,1]])),axis=0)
    #         matrix= np.matmul(R,M)
        
    #         img = cv2.warpPerspective(img,matrix, (cols,rows),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    #         if len(img.shape)==2:
    #             img = np.expand_dims(img, 2)
            
    #         if not config.method == 'pretraining':
    #             mask = cv2.warpPerspective(mask,matrix, (cols,rows),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)
    
    
    # in_size=img.shape
    # out_size=[config.patch_size,config.patch_size]
        
    
    

    # r1=int((in_size[0]-out_size[0])/2)
    # r2=int((in_size[1]-out_size[1])/2)
        
    # r=[r1,r2]
    
    
    # img=img[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1],:]
    # if not config.method == 'pretraining':
    #     mask=mask[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
    
    
    
    # if rand()>config.p:
    #     multipy=config.multipy
    #     multipy=1+rand()*multipy
    #     if rand()>0.5:
    #         img=img*multipy
    #     else:
    #         img=img/multipy
       
    # if rand()>config.p:
    #     add=config.add     
    #     add=(1-2*rand())*add
    #     img=img+add
    
    
    # if img.shape[2]>1:
    #     for slice_num in range(img.shape[2]):
            
    #         slice_ = img[:,:,slice_num]
            
    #         if rand()>config.p:
    #             multipy=0.1 
    #             multipy=1+rand()*multipy
    #             if rand()>0.5:
    #                 slice_=slice_*multipy
    #             else:
    #                 slice_=slice_/multipy
               
    #         if rand()>config.p:
    #             add=0.1     
    #             add=(1-2*rand())*add
    #             slice_=slice_+add
                
    #         img[:,:,slice_num] = slice_
    
    
    
    
    
    # if rand()>config.p:
    #     bs_r=(-config.sharp,config.blur)
    #     r=1-2*rand()
    #     if r<=0:
    #         par=bs_r[0]*r
    #         img=img-par*laplace(img)
    #     if r>0:
    #         par=bs_r[1]*r
    #         img=gaussian_filter(img,par)
    
    
    
    return img,mask 




class Dataset(data.Dataset):


    def __init__(self, names,augment,config,data_type=None):
       
        self.names = names
        self.augment = augment
        self.config = config
        self.data_type = data_type
        
        self.names = self.names
        
        self.N = len(self.names)
        
        if not self.config.method == 'pretraining':
            
            self.h5data_file = config.data_path + '/' +  'dataset.hdf5'
            
        else:
            
            self.h5data_file = config.data_path + '/' +  'dataset_pretrain.hdf5'
        
        self.h5data = None

    def __len__(self):
        return self.N*self.config.multiply_dataset


    def __getitem__(self, idx):
        
        if  self.h5data is None:
            self.h5data = h5py.File(self.h5data_file, 'r')

        idx = idx % self.N

        name = self.names[idx]
        

        if not self.config.method == 'pretraining':
            
            tmp = name.split('/')
            mask = self.h5data[tmp[0]][tmp[1]][:,:]
            mask = np.transpose(mask,(1,0))
            mask = (mask>0).astype(np.uint8)
            name_tmp = '_'.join(name.split('_')[:-1])
            name_tmp = name_tmp.replace('Vessels','Images').replace('Disc','Images').replace('Cup','Images')
            
        else:
            mask = None
            name_tmp = name
            
        
        tmp = name_tmp.split('/') 
        img = self.h5data[tmp[0]][tmp[1]][:,:,:]
        img = np.transpose(img,(2,1,0))
        img = img.astype(np.float64)/255 - 0.5
               
            
        # if self.augment:
        #     in_size=img.shape
        #     out_size=[self.config.patch_size*2,self.config.patch_size*2]
        # else: 
        #     in_size=img.shape
        #     out_size=[self.config.patch_size,self.config.patch_size]
            
        in_size=img.shape
        out_size=[self.config.patch_size,self.config.patch_size] 
        
        
    
        r1=torch.randint(in_size[0]-out_size[0],(1,1)).view(-1).numpy()[0]
        r2=torch.randint(in_size[1]-out_size[1],(1,1)).view(-1).numpy()[0]
        r=[r1,r2]
        
        
        img=img[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1],:]
        if not self.config.method == 'pretraining':
            mask=mask[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
        
            
                
  
        
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
        
        

        
        if self.config.clahe:
            
            img = np.floor((img + 0.5) * 255 ).astype(np.uint8)
            
            if img.shape[2]==1:
                
                
                clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip,tileGridSize=(self.config.clahe_grid,self.config.clahe_grid))
                img = clahe.apply(img[:,:,0])
                img = np.expand_dims(img,2)

            else:
                

                
                planes = cv2.split(img)
        
                clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip,tileGridSize=(self.config.clahe_grid,self.config.clahe_grid))
                
                planes[0] = clahe.apply(planes[0])
                planes[1] = clahe.apply(planes[1])
                planes[2] = clahe.apply(planes[2])
                
                img = cv2.merge(planes)
                
            img = img.astype(np.float64)/255 - 0.5
                
                
        img = img.astype(np.float32) 
            
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
                
                
                
            block_types = ['del' for _ in range(self.config.pretrain_num_blocks)]
                

            used = np.zeros(img_shape[:2])
            for k,block_type in enumerate(block_types):
                

                block_sizex = int(np.ceil(rand()*self.config.pretrain_max_block_size))
                block_sizey = int(np.ceil(rand()*self.config.pretrain_max_block_size))
  
                posx = int(np.round(rand()*(img_shape[0]-block_sizex)))
                posy = int(np.round(rand()*(img_shape[1]-block_sizey)))
                
                # if np.sum(used[posx:posx+block_sizex,posy:posy+block_sizey])==0:
                if True:
                    
                    used[posx:posx+block_sizex,posy:posy+block_sizey] = 1
                    
                    
            
                    std_tmp = self.config.pretrain_std
                    mean_tmp = self.config.pretrain_mean

                    # block = torch.randn([block_sizex,block_sizey,img.shape[2]]).numpy()*std_tmp + mean_tmp
                    block = torch.randn([block_sizex,block_sizey,img.shape[2]]).numpy()*std_tmp + mean_tmp
                
                    img[posx:posx+block_sizex,posy:posy+block_sizey,:] = block
                    
                    

        if len(mask.shape)==2:
            mask = mask.reshape([mask.shape[0],mask.shape[1],1])
        mask=torch.from_numpy(np.transpose(mask,(2,0,1)).astype(np.float32))
        img=torch.from_numpy(np.transpose(img,(2,0,1)).astype(np.float32))
        
        
        return img,mask







if __name__ == "__main__":

    
    
    
    from config import Config    
    
    config = Config()
    config.method = 'segmentation'
    # config.method = 'pretraining'
    
    
    
    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=42)
    
    train_generator = Dataset(data_split['train'],augment=True,config=config)
    # train_generator = Dataset(data_split['pretrain_train'],augment=True,config=config)
    
    train_generator = data.DataLoader(train_generator,batch_size=1,num_workers= 0, shuffle=True,drop_last=True)
    
    start = time.time()
    for it,(img,mask) in enumerate(train_generator):
        
        # if it%10 == 0:
        #     end = time.time()
        #     print(end - start)
        #     start = time.time()
        
        
        plt.imshow(np.transpose(img[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        plt.show()
        plt.imshow(np.transpose(mask[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        plt.show()
        break
    