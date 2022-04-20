import numpy as np
import os
import pydicom as dcm

import matplotlib.pyplot as plt

# from Utilities import size_nii




def CreateDataset_dcm(path_data, text1, text2):
    data_list_tr = []
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        if pat_name.find(text1)>=0:
            for _,file in enumerate(f):
                if file.find('_va')>0:
                    if file.find(text2)>=0:
                        path_maps=[]
                        
                        path_mask = os.path.join(path_data, pat_name, file)
                        name = file.replace('_va.dcm','')
                        
                        path_maps.append( path_mask.replace('va','R') )
                        path_maps.append( path_mask.replace('va','G') )
                        path_maps.append( path_mask.replace('va','B') )
                        

                        path_maps.append( path_mask.replace('va','ves') )

                        # sizeData = size_nii( path_maps_1 )
    
                        # if len(sizeData)==2:
                        #     sizeData = sizeData + (1,)
                        # # print(sizeData)
                        
                        data_list_tr.append( {'img_path': path_maps,
                                              'mask_path': path_mask,
                                              'pat_name': pat_name,
                                              'file_name': name } )
            
    return data_list_tr
