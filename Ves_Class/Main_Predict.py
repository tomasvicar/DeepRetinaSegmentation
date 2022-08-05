import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import Loaders
import Network as Network
import cv2
# import Network_v9 as Network

torch.cuda.empty_cache()   

version_load = 'net_v0_0_9'
net = torch.load("/home/chmelikj/Documents/chmelikj/Ophtalmo/DeepRetinaSegmentation/Ves_Class/Models/" + version_load + ".pt")


path_save = '/home/chmelikj/Documents/chmelikj/Ophtalmo/Data/results_predict_UBMI_25N_' + version_load

net = net.cuda()

## Data OPTHALMO
path_data = '/home/chmelikj/Documents/chmelikj/Ophtalmo/Data/data_preprocessed_dicom_25N_UBMI'
data_list = Loaders.CreateDataset_dcm_predict(os.path.normpath( path_data ), '','')

data_list_1_predict = data_list
        
net.train(mode=False)

for num in range(0,len(data_list_1_predict)):
    with torch.no_grad():
        resVal, Imgs = Network.Predict(data_list_1_predict[num], net, TrainMode=False)   

        torch.cuda.empty_cache() 
        resVal = resVal.detach().cpu().numpy()
        Imgs = Imgs.detach().cpu().numpy()
        
        resValClass = np.zeros(resVal[0,0,:,:].shape)
        resValClass[resVal[0,1,:,:]>resVal[0,2,:,:]] = 1
        resValClass[resVal[0,1,:,:]<resVal[0,2,:,:]] = 2
        resValClass[Imgs[0,3,:,:]==0] = 0
        
        plt.figure
        plt.imshow(resValClass, cmap='jet')
        plt.show()
        
        resValClassOrig = cv2.resize(resValClass, dsize=(data_list_1_predict[num]['orig_size'][1], data_list_1_predict[num]['orig_size'][0]), interpolation=cv2.INTER_NEAREST)
        
        full_path_save = path_save + '/Sada_01/' + data_list_1_predict[num]['file_name'] + '/ImageAnalysis/VesselsClass/'
        isExist = os.path.exists(full_path_save)
        if not isExist:
          os.makedirs(full_path_save)
        
        plt.imsave(path_save + '/Sada_01/' + data_list_1_predict[num]['file_name'] + 
                   '/ImageAnalysis/VesselsClass/' +
                   data_list_1_predict[num]['file_name'] +
                   '_VA_classification_whole.png', resValClassOrig)