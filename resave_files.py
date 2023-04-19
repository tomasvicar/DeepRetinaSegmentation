from glob import glob
import os
import shutil


# input_path = '../spocitane_trasformace_all/Sada_01'
# output_path = '../final_multimodal_results/Sada_01'

input_path = r'C:\Data\Vicar\retina_vessels_segmentation\Sada_02'
output_path = r'C:\Data\Vicar\retina_vessels_segmentation\trasfromed_segmentation_02\Sada_01'


sub_path = '/ImageAnalysis/VesselsSeg/'


to_resaves = []
resaved_names = []

to_resaves.append('')
resaved_names.append('_segmentation_whole.png')



filenames = glob(input_path + '/**/*_segmentation_norm_hrf.png',recursive=True)


for file_num, filename in enumerate(filenames):
    
    
    filename_save =  filename.replace(input_path,'')
    
    filename_save = os.path.normpath(filename_save)
    
    filename_save = output_path + os.sep + filename_save.split(os.sep)[1] + sub_path + os.path.split(filename_save)[1]
    
    
    
    filename_save = filename_save.replace('_segmentation_norm_hrf.png','') + '_segmentation_whole.png'
    
    if not os.path.exists(os.path.split(filename_save)[0]):
        os.makedirs(os.path.split(filename_save)[0]) 
    
    shutil.copyfile(filename, filename_save)


