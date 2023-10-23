from DataSpliter import DataSpliter
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm
from generate_dataset_json import generate_dataset_json

def convert_to_nnUNetv2(dataset_name, data_path, segmentation_type):
    path_imagesTr = os.path.join('../nnUNet_raw', dataset_name, 'imagesTr')
    path_imagesTs = os.path.join('../nnUNet_raw', dataset_name, 'imagesTs')
    path_labelsTr = os.path.join('../nnUNet_raw', dataset_name, 'labelsTr')
    path_labelsTs = os.path.join('../nnUNet_raw', dataset_name, 'labelsTs')

    os.makedirs(path_imagesTr, exist_ok=True)
    os.makedirs(path_labelsTr, exist_ok=True)
    os.makedirs(path_imagesTs, exist_ok=True)
    os.makedirs(path_labelsTs, exist_ok=True)

    data_spliter = DataSpliter(data_path,
                                train_valid_test_frac=[0.7, 0.1, 0.2],
                                mask_type_use=[DataSpliter.VESSEL_CLASS],
                                seed=42
                                )
    dataset_dict = data_spliter.split_data()
    num_train = 0
    for key in tqdm(dataset_dict):
        split = dataset_dict[key]['split']
        with h5py.File(data_path, 'r') as h5f:
            group = h5f[key]
            image = np.array(group['img'])
            label = np.array(group[segmentation_type])
            if split == 'train':
                path_save_img = path_imagesTr
                path_save_lbl = path_labelsTr
                num_train = num_train+1
            elif split == 'test':
                path_save_img = path_imagesTs
                path_save_lbl = path_labelsTs
            else:
                continue
            io.imsave(os.path.join(path_save_img, key + '_0000.png'), image.astype(np.uint8), check_contrast=False)
            io.imsave(os.path.join(path_save_lbl, key + '.png'), label.astype(np.uint8), check_contrast=False)
    return num_train

if __name__ == '__main__':
    dataset_name = 'Dataset002_VOVesselsClass'
    data_path = '../data_25_normalized.hdf5'
    segmentation_type = 'ves_class'
    num_train = convert_to_nnUNetv2(dataset_name, data_path, segmentation_type)
    generate_dataset_json(os.path.join('../nnUNet_raw', dataset_name),
                          {0: 'R', 1: 'G', 2: 'B'},
                          {'background': 0, 'vessel': 1},
                          num_training_cases=num_train,
                          file_ending='.png',
                          dataset_name=dataset_name,
                          description='25_' + segmentation_type + '_normalized_seed_42')
