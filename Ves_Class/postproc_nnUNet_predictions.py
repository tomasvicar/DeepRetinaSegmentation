import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage.io import imread
from Utilities import imfusion
import nibabel as nib
import pandas as pd
from skimage.transform import resize

path_save = '/home/chmelikj/Documents/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task535_Ophtalmo/predictTs_postproc/'
path_res_info = '/media/chmelikj/DATA/DeepRetinaSegmentationData/data_preprocessed_dicom_35nnUNet/UBMI/'

## Data OPTHALMO
path_data = '/home/chmelikj/Documents/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task535_Ophtalmo/predictTs/'
path_data_orig = '/media/chmelikj/DATA/DeepRetinaSegmentationData/Sada_01/'
data_list = os.listdir(path_data)
info_list = os.listdir(path_res_info)

for num, pat_name in enumerate(data_list):
    pat_path = os.path.join(path_data, pat_name)
    resVal = nib.load(pat_path)
    resVal = np.array(resVal.dataobj)[:,:,0]
    
    plt.figure
    plt.imshow(resVal, cmap='jet')
    plt.show()
    
    df = pd.read_csv(os.path.join(path_res_info, pat_name.replace('.nii.gz','_orig_size_info.csv')), header=None).to_numpy()
    
    resValOrig = resize(resVal, output_shape=df[0][0:2], order=0, anti_aliasing=False)
    
    full_path_save = path_save + 'Sada_01/' + pat_name.replace('.nii.gz', '') + '/ImageAnalysis/VesselsClass/'
    isExist = os.path.exists(full_path_save)
    if not isExist:
      os.makedirs(full_path_save)
    else:
        print('Scan: {} exist! ({}/{})'.format(pat_name.replace('.nii.gz', ''), num+1, len(data_list)))
        continue
    
    resValOrig[resValOrig==1] = 128
    resValOrig[resValOrig==2] = 255
    imsave(full_path_save + pat_name.replace('.nii.gz', '') + '_VA_classification_nnUNet.png', resValOrig.astype('uint8'))
    
    full_path_orig = path_data_orig + pat_name.replace('.nii.gz', '') + '/' + pat_name.replace('.nii.gz', '')[:-5] + 'L.jpg'
    isExist = os.path.exists(full_path_orig)
    if isExist:
        Orig = imread(full_path_orig)
        
    full_path_orig = path_data_orig + pat_name.replace('.nii.gz', '') + '/' + pat_name.replace('.nii.gz', '')[:-5] + 'L.JPG'
    isExist = os.path.exists(full_path_orig)
    if isExist:
        Orig = imread(full_path_orig)
        
    resValFusion = imfusion(Orig, resValOrig)
    imsave(full_path_save + pat_name.replace('.nii.gz', '') + '_VA_classification_nnUNet_fusion.png', resValFusion.astype('uint8'))
    print('Scan: {} done! ({}/{})'.format(pat_name.replace('.nii.gz', ''), num+1, len(data_list)))