## U-net
 # for version 2

import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import pydicom as dcm
import Utilities as Util
import random
from torch.nn import init

# import matplotlib.pyplot as plt
# from PIL import Image 



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate')
        # self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate')
        self.BN    = nn.BatchNorm2d(in_ch)
    
    def forward(self, x):
        x = self.conv1(self.BN(x))
        x = self.relu(x)    # for v7_0_0
        res = x
        # x = self.conv3(self.conv2(x))
        x = self.conv2(x)
        return self.relu(x) + res
        # return self.relu(x)


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
        self.relu       = nn.ReLU()
    
    def forward(self, x):
        ftrs = []
        m, s = torch.mean(x,(2,3)), torch.std(x,(2,3))
        x = (x - m[:,:,None, None]) / s[:,:,None,None]
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class BottleNeck(nn.Module):
    def __init__(self, chs=(1024,1024) ):
        super().__init__()
        self.conv1x1_1 =  nn.Conv2d(chs[0], chs[1], 1, padding=0, padding_mode='replicate')     
        # self.conv1x1_2 =  nn.Conv2d(chs[0], chs[1], 1, padding=0, padding_mode='replicate')     
        # self.DP =  nn.Dropout(p=0.5)
    def forward(self, x):
        
        # return self.conv1x1_2(self.DP(self.conv1x1_1(x)))
        return self.conv1x1_1(x)



class Net(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, head=(128)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.bottleneck  = BottleNeck((enc_chs[-1],enc_chs[-1]))
        self.decoder     = Decoder(dec_chs)
        # self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.head1       = nn.Conv2d(dec_chs[-1], head, 3, padding=1)
        self.head2       = nn.Conv2d(head, num_class, 1, padding=0)

        self.relu       = nn.ReLU()
        # self.DP_H =  nn.Dropout(p=0.5)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        OutBN = self.bottleneck(enc_ftrs[::-1][0])
        out      = self.decoder(OutBN, enc_ftrs[::-1][1:])
        out      = self.head2(  self.head1( out )  )     # for v7_0_0
        # out      =  self.head( out ) 
        
        out = F.interpolate(out, (x.size(2),x.size(3)) )

        return out
    
  
    
class Training():  
    
    def straightForward(data_list, net, params, TrainMode=True, Contrast=False): 

        net.train(mode=TrainMode)
        batch = len(data_list)
        vel = params[0]
        # vel = 256
        num_ch = len(data_list[0]['img_path'])
        
        Imgs = torch.tensor(np.zeros((batch,num_ch,vel,vel) ), dtype=torch.float32)
        Masks = torch.tensor(np.zeros((batch,1,vel,vel) ), dtype=torch.float32)
        
        for b in range(0,batch):
            
            img_paths = data_list[b]['img_path']
            mask_path = data_list[b]['mask_path']
            dc_pos = data_list[b]['dc_pos']
            
    
            augm_params=[]; 
            augm_params.append({'Output_size': params[0],
                            'Crop_size': random.randint(params[1],params[2]),
                            'Angle': random.randint(params[3],params[4]),
                            'Transl': (random.randint(params[5],params[6]),random.randint(params[7],params[8])),
                            'Scale': random.uniform(params[9],params[10]),
                            'Flip':  np.random.random()>0.5
                            })
            
            augm = random.uniform(0, 1)>=0.3
            # augm = True
            
            t=0; sl=0
            for ch, ch_path in enumerate(img_paths):
                # print('{:s}'.format(ch_path))
                dataset = dcm.dcmread(ch_path)
                img = dataset.pixel_array.astype(dtype='float32')
                                
                if not augm:
                    # central crop and padding to same size
                    # img = Util.resize_with_padding(img,(vel,vel))
                    
                    # optical disc position crop and padding to same size
                    img = Util.resize_with_padding_dc(img, (vel,vel), dc_pos)
                img = np.expand_dims(img, 0).astype(np.float32)
                img = torch.tensor(img)
                
                if  augm:
                    img = Util.augmentation3(img, augm_params, dc_pos)

                Imgs[b,ch,:,:] = img
                
            dataset = dcm.dcmread(mask_path)
            mask = dataset.pixel_array
            # mask = Util.resize_with_padding(mask,(vel,vel))
            if not augm:
                mask = Util.resize_with_padding_dc(mask, (vel,vel), dc_pos) 
                
            mask = np.expand_dims(mask, 0).astype(np.float32)
            mask = torch.tensor(mask)
            
            if  augm:
                mask = Util.augmentation3(mask, augm_params, dc_pos)
                # mask = mask>0.5 
                
            Masks[b,0,:,:] = torch.round(mask)

        res = net( Imgs.cuda() )
        
        res_vec = torch.reshape(res, (batch, res.size(dim=1), res.size(dim=2)*res.size(dim=3)))
        res_vec = res_vec[:,(1,2),:]
        res_pp = torch.softmax(res_vec,dim=1)
        res_pp0 = torch.reshape(res_pp[:,0,:], [-1])
        res_pp1 = torch.reshape(res_pp[:,1,:], [-1])
        
        loss = nn.CrossEntropyLoss()
        
        # output = loss( res, torch.squeeze(Masks).long().cuda() )
        # output = loss( res[:,:,:,:], torch.squeeze(Masks[:,:,:,:]).long().cuda() )
        Masks_vec = torch.reshape(torch.squeeze(Masks[:,:,:,:]).long().cuda(), (batch, res_pp.size(dim=2)))
        Masks_vec = torch.reshape(Masks_vec, [-1])
        
        res_pp0 = res_pp0[Masks_vec>0]
        res_pp1 = res_pp1[Masks_vec>0]
        Masks_vec = Masks_vec[Masks_vec>0]
        Masks_vec = Masks_vec-1
        
        res_vec = torch.cat((torch.unsqueeze(res_pp0, 0), torch.unsqueeze(res_pp1, 0)), 0)
        output = loss(torch.unsqueeze(res_vec, 0), torch.unsqueeze(Masks_vec, 0))
        # output = loss( res[:,:,:,:], torch.squeeze(Masks[:,:,:,:]).long().cuda() )
 
        # loss = nn.MultiLabelSoftMarginLoss()
        # output = loss(res, Util.ToOneHot(Masks, numClass=3 ).cuda())
        
        return output, res, Imgs, Masks
    
   
    
   
   