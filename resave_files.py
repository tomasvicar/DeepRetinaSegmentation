from glob import glob
import os
import shutil


# input_path = '../spocitane_trasformace_all/Sada_01'
# output_path = '../final_multimodal_results/Sada_01'

input_path = r'C:\Data\Vicar\retina_vessels_segmentation\Sada_01\Sada_01'
output_path = r'C:\Data\Vicar\retina_vessels_segmentation\trasfromed_segmentation\Sada_01'


sub_path = '/ImageAnalysis/VesselsSeg/'


to_resaves = []
with_names = []

to_resaves.append('/**/*_avg_registered_nocorrupted.png')
with_names.append(True)

to_resaves.append('/**/*_disk_detection.png')
with_names.append(True)

to_resaves.append('/**/first_try_best/init_trasform_parameters.json')
with_names.append(False)

to_resaves.append('/**/first_try_best/transformation.npy'  )
with_names.append(False)
       

for to_resave, with_name in zip(to_resaves,with_names):

    filenames = glob(input_path + to_resave,recursive=True)
    
    
    for file_num, filename in enumerate(filenames):
        
        
        filename_save =  filename.replace(input_path,'')
        
        filename_save = os.path.normpath(filename_save)
        
        if not with_name:
            filename_save = output_path + os.sep + filename_save.split(os.sep)[1] + sub_path + filename_save.split(os.sep)[1] + '_' + os.path.split(filename_save)[1]
        else:
            filename_save = output_path + os.sep + filename_save.split(os.sep)[1] + sub_path + os.path.split(filename_save)[1]
        
        if not os.path.exists(os.path.split(filename_save)[0]):
            os.makedirs(os.path.split(filename_save)[0]) 
        
        shutil.copyfile(filename, filename_save)


