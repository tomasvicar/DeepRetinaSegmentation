import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import random
import pickle
import copy

import Utilities as Util
import Loaders
import Network as Network
# import Network_v9 as Network


params = {'optim_lr':       0.0000002,
          'optim_L2':       0.000001,
          'optim_type':     'Adam',
          'loss_type':      'CrossEntropy',
          'batchTr':        6,
          'batchVal':      20,
          'sch_step_size':200,
          'sch_gamma':    0.1,
          'sch_type':       'StepLR',
          'hdm_sigma':      0.7,
          'lambda_Train':   1.0,
          'num_ite':       50,
          'num_epch':    1000,
          'aug_crop_size':450,
          'aug_rot_angle': 10,
          'aug_trans_%':  30,
          'aug_scale_%':   15,
          'aug_flip_p':     0.5,
          'augTr_p':        0.7,
          'augVal_p':       0.0,
          'init_weights':   'xavier',
          'init_gain':      0.02,
          'tr_val_ratio':   0.80,
          'rng_seed':    2022,
          'net_type':       'UNet',
          'net_enc_chs':    (4,32,64,128,256),
          'net_dec_chs':    (256,128,64,32),
          'net_head':       (128),
          'num_class':      3,
          'dataset':        'data_preprocessed_dicom_25N',
          'version':        'v0_0_9'}

np.random.seed(params['rng_seed'])

batchTr = int(np.round(params['batchTr']))
step_size = int(np.round(params['sch_step_size']))
num_ite = int(np.round(params['num_ite']))
 
torch.cuda.empty_cache()   

# if params['net_type']=='UNet':
#     net = Network.Net(enc_chs=params['net_enc_chs'], dec_chs=params['net_dec_chs'], head=params['net_head'], num_class=params['num_class'])
# elif params['net_type']=='AttUNet':
#     net = Network.AttU_Net(img_ch=1,output_ch=1)
    
# Network.init_weights(net,init_type= params['init_weights'], gain=params['init_gain'])

version_load = 'net_v0_0_8'
net = torch.load("/home/chmelikj/Documents/chmelikj/Ophtalmo/DeepRetinaSegmentation/Ves_Class/Models/" + version_load + ".pt")

net = net.cuda()
if params['optim_type']=='Adam':
    optimizer = optim.Adam(net.parameters(), lr=params['optim_lr'], weight_decay=params['optim_L2'])
elif params['optim_type']=='SGD':
    optimizer = optim.SGD(net.parameters(), lr=params['optim_lr'], weight_decay=params['optim_L2'], momentum= 0.8)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['sch_step_size'], gamma=params['sch_gamma'], verbose=False)

data_list_1_train=[];data_list_1_val=[];

## Data OPTHALMO
path_data = '/home/chmelikj/Documents/chmelikj/Ophtalmo/Data/' + params['dataset']
data_list = Loaders.CreateDataset_dcm_with_DC(os.path.normpath( path_data ), '','')

b = int(len(data_list)*params['tr_val_ratio'])
data_list_1_train = data_list_1_train + data_list[1:b+1]
data_list_1_val = data_list_1_val + data_list[b:]


lossTr=[]; lossVal=[];
accTr=[]; accVal=[];

D1 = np.zeros((len(data_list_1_train),2))
D1[:,0] = np.arange(0,len(data_list_1_train))


for epch in range(0,params['num_epch']):
    
    mu1, sigma1 = len(data_list_1_train)/10 , params['hdm_sigma']*len(data_list_1_train)
    
    net.train(mode=True)

    lossTr1=[]; lossVal1=[];
               
    # for num_ite in range(0,len(data_list_1_train)-batchTr-1, batchTr):
    # for num_ite in range(0,len(data_list_1_train)/batchTr):
    for n_ite in range(0,num_ite):
      
        Indx_Sort = Util.rand_norm_distrb(batchTr, mu1, sigma1, [0,len(data_list_1_train)]).astype('int')
        Indx_Orig = D1[Indx_Sort,0].astype('int')
        sub_set = copy.deepcopy(list(map(data_list_1_train.__getitem__, Indx_Orig)))
        
        params_aug_tr = (params['aug_crop_size'], # output size
                         params['aug_crop_size'], # crop size x
                         params['aug_crop_size'], # crop size y
                         -params['aug_rot_angle'], # max rotation angle CCW
                         params['aug_rot_angle'], # max rotation angle CW
                         int(-params['aug_crop_size']/100*params['aug_trans_%']), # max translation left (% of crop)
                         int(params['aug_crop_size']/100*params['aug_trans_%']), # max translation right (% of crop)
                         int(-params['aug_crop_size']/100*params['aug_trans_%']), # max translation up (% of crop)
                         int(params['aug_crop_size']/100*params['aug_trans_%']), # max translation down (% of crop)
                         (100-params['aug_scale_%'])/100, # max shrink %
                         (100+params['aug_scale_%'])/100,
                         params['aug_flip_p'], # probaility of fliping
                         params['augTr_p']) # probaility augmentation
        
        loss_train, res, _, Masks = Network.Training.straightForward(sub_set, net, params_aug_tr, TrainMode=True, Contrast=False)
        
        metric  = loss_train.detach().cpu().item() 
        
        lossTr1.append( metric )
        
        D1[np.array(Indx_Sort),1] = np.array(10-metric)

        D1 = D1[D1[:, 1].argsort()]
        

    ## backF - training
        net.train(mode=True)
        if epch>0:
            
            loss = params['lambda_Train']*loss_train
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
            optimizer.step()
    
    # pd = norm(mu1,sigma1)
    # plt.figure()
    # plt.plot(D1[:,1])
    # y = pd.pdf([np.linspace(0,np.size(D1,0),np.size(D1,0))]).T
    # plt.plot(y/y.max())
    # plt.ylim([0.0, 1.1])
    # plt.show()

    resCPU = res.detach().cpu().numpy()
    MasksCPU = Masks.detach().cpu().numpy()
    resClass = np.zeros(resCPU[:,0,:,:].shape)
    resClass[resCPU[:,1,:,:]>resCPU[:,2,:,:]] = 1
    resClass[resCPU[:,1,:,:]<resCPU[:,2,:,:]] = 2
    resClass[MasksCPU[:,0,:,:]==0] = 0
    
    plt.figure
    plt.imshow(MasksCPU[0,0,:,:], cmap='jet')
    plt.show()
    plt.figure
    plt.imshow(resClass[0,:,:], cmap='jet')
    plt.show()
    
    if epch>0:
        scheduler.step()
        
    net.train(mode=False)
   
    ### validation
    params_aug_val = (params['aug_crop_size'], # output size
                     params['aug_crop_size'], # crop size x
                     params['aug_crop_size'], # crop size y
                     -params['aug_rot_angle'], # max rotation angle CCW
                     params['aug_rot_angle'], # max rotation angle CW
                     int(-params['aug_crop_size']/100*params['aug_trans_%']), # max translation left (% of crop)
                     int(params['aug_crop_size']/100*params['aug_trans_%']), # max translation right (% of crop)
                     int(-params['aug_crop_size']/100*params['aug_trans_%']), # max translation up (% of crop)
                     int(params['aug_crop_size']/100*params['aug_trans_%']), # max translation down (% of crop)
                     (100-params['aug_scale_%'])/100, # max shrink %
                     (100+params['aug_scale_%'])/100,
                     params['aug_flip_p'], # probaility of fliping
                     params['augVal_p']) # probaility augmentation
    
    batchVal = params['batchVal']
    random.shuffle(data_list_1_val)
    # for num in range(0,len(data_list_1_val), batchVal):
    for num in range(0,1):
        sub_set = copy.deepcopy(data_list_1_val[num:num+batchVal])
        
        with torch.no_grad():
            
            loss_val, resVal, ImgsVal, MasksVal = Network.Training.straightForward(sub_set, net, params_aug_val, TrainMode=False, Contrast=False)   
        
        lossVal1.append(loss_val.detach().cpu().item() )

    torch.cuda.empty_cache() 

    lossTr.append(np.mean(lossTr1))
    lossVal.append(np.mean(lossVal1))

    resVal = resVal.detach().cpu().numpy()

    # print('image: {:s}'.format(sub_set[0]['file_name']))
    # plt.figure
    # # plt.imshow(ImgsVal[0,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(resVal[0,1,:,:], cmap='jet', alpha=0.5)
    # plt.imshow(resVal[0,2,:,:], cmap='jet', alpha=0.5)
    # plt.show()
    
    resValClass = np.zeros(resVal[:,0,:,:].shape)
    resValClass[resVal[:,1,:,:]>resVal[:,2,:,:]] = 1
    resValClass[resVal[:,1,:,:]<resVal[:,2,:,:]] = 2
    resValClass[MasksVal[:,0,:,:]==0] = 0
    
    resValClassVec = np.reshape(resValClass, [-1])
    MasksValVec = np.reshape(MasksVal.detach().cpu().numpy(), [-1])
    resValClassVec = resValClassVec[MasksValVec>0]
    MasksValVec = MasksValVec[MasksValVec>0]
    
    accVal1 = Util.acc_metric(resValClassVec, MasksValVec)
    
    accVal.append(accVal1)
    
    # plt.figure
    # plt.imshow(resVal[0,0,:,:], cmap='jet')
    # plt.show()
    # plt.figure
    # plt.imshow(resVal[0,1,:,:], cmap='jet')
    # plt.show()   
    # plt.figure
    # plt.imshow(resVal[0,2,:,:], cmap='jet')
    # plt.show()
    plt.figure
    plt.imshow(MasksVal[0,0,:,:], cmap='jet')
    plt.show()
    plt.figure
    plt.imshow(resValClass[0,:,:], cmap='jet')
    plt.show()
    # plt.figure
    # plt.imshow(ImgsVal[0,3,:,:], cmap='jet')
    # plt.show()
   
    
    plt.figure()
    plt.plot(lossTr,label='Joint Train')
    plt.plot(lossVal,label='Joint Val')
    plt.plot(accVal,label='Accuracy Val')

    plt.ylim([0.3, 1.0])
    plt.legend()
    plt.show()
    
    accValBest = max(accVal)
    if accVal[-1] >= accValBest:
        torch.save(net, 'Models/net_' + params['version'] + '.pt')
    
        file_name = "Models/net_" + params['version'] + ".pkl"
        open_file = open(file_name, "wb")
        pickle.dump([params, {'lossTr': lossTr, 'lossVal': lossVal, 'accVal': accVal, 'epoch': epch}], open_file)
        open_file.close()
