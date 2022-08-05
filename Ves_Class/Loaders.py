import numpy as np
import os
import pydicom as dcm
import pandas as pd
import csv
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

# create dataset in dicom containing info about centre of optical disc
def CreateDataset_dcm_with_DC(path_data, text1, text2):
    data_list_tr = []
    p = os.listdir(path_data)
    dc_path = os.path.split(path_data)
    dc_path = os.path.join(dc_path[0],'disc_coordinates_'+dc_path[1].split('_')[-1])
    dc_files = os.listdir(dc_path)
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
                        
                        matching = [s for s in dc_files if pat_name in s]
                        my_data = pd.read_csv(os.path.join(dc_path, matching[0]), sep=',')
                        matching = my_data.loc[my_data['name'] == os.path.basename(path_mask.replace('_va.dcm',''))]
                        dc_pos = matching['x-coordinates'].values.tolist()
                        dc_pos.append(matching['y-coordinates'].values.tolist()[0])
                        
                        data_list_tr.append( {'img_path': path_maps,
                                              'mask_path': path_mask,
                                              'pat_name': pat_name,
                                              'file_name': name,
                                              'dc_pos': dc_pos} )
            
    return data_list_tr

# create dataset in dicom for prediction
def CreateDataset_dcm_predict(pat_name, text1, text2):
    data_list_te = []
    f = os.listdir(pat_name)
    if pat_name.find(text1)>=0:
        for _,file in enumerate(f):
            if file.find('_R')>0:
                if file.find(text2)>=0:
                    path_maps=[]
                    
                    path_img = os.path.join(pat_name, file)
                    name = file.replace('_R.dcm','')
                    
                    path_maps.append( path_img )
                    path_maps.append( path_img.replace('R','G') )
                    path_maps.append( path_img.replace('R','B') )
                    path_maps.append( path_img.replace('R','ves') )
                    
                    file = open(path_img.replace('R.dcm','orig_size_info.csv'))
                    csvreader = csv.reader(file)
                    for row in csvreader:
                            orig_size = [int(x) for x in row]
                    file.close()
                    
                    data_list_te.append( {'img_path': path_maps,
                                          'pat_name': pat_name,
                                          'file_name': name,
                                          'orig_size': orig_size} )
            
    return data_list_te