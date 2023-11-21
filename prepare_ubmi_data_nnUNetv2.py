from PIL import Image
from utils.local_contrast_and_clahe import local_contrast_and_clahe
from datasets_loaders.LoaderGeneric import LoaderGeneric
import numpy as np
import os


def prepare_ubmi_data_nnunetv2(images_path, output_path):
    files = os.listdir(images_path)
    for file in files:
        image = Image.open(os.path.join(images_path, file))
        image_array = np.asarray(image)
        image_preprocessed = local_contrast_and_clahe(image_array, LoaderGeneric.get_fov_auto(image_array), 25)
        image_preprocessed = Image.fromarray(image_preprocessed)
        image_preprocessed_name = os.path.splitext(file)[0] + '_0000.png'
        image_preprocessed.save(os.path.join(output_path, image_preprocessed_name))


if __name__ == "__main__":
    prepare_ubmi_data_nnunetv2(r'D:\DeepRetinaSegmentationData\UBMI_test_data\Vessel_masks_labeled\Fundus_Images',
                      r'D:\DeepRetinaSegmentationData\UBMI_test_data\Vessel_masks_labeled\Fundus_Images_Preprocessed')
