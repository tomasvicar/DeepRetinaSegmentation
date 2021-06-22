from skimage.io import imread
import numpy as np


from split_data import DataSpliter






data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=42)

means = []
stds = []


for name in data_split['train']:
    
    name_tmp = '_'.join(name.split('_')[:-1]) + '.png'
    name_tmp = name_tmp.replace('Vessels','Images').replace('Disc','Images').replace('Cup','Images')

    img = imread(name_tmp)
    img = img.astype(np.float64)
    img = (img/255)-0.5
    
    name_tmp = '_'.join(name.split('_')[:-1]) + '_fov.png'
    name_tmp = name_tmp.replace('Vessels','Fov').replace('Disc','Fov').replace('Cup','Fov')
    fov =  imread(name_tmp)>0
    fov = np.stack((fov,fov,fov),axis=2)

    means.append(np.mean(img[fov]))
    stds.append(np.std(img[fov]))
    
    
    
print(np.mean(means))
print(np.mean(stds))