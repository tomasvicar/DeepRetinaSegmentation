import imagesize
import os
import cv2
import numpy as np


def resize_segmentation(input_path, orig_path, output_path):
    files = os.listdir(orig_path)
    for file in files:
        width, height = imagesize.get(os.path.join(orig_path, file))
        image = cv2.imread(os.path.join(input_path, os.path.splitext(file)[0] + '.png'), cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_path, os.path.splitext(file)[0] + '.png'), np.round(image_resized))

if __name__ == "__main__":
    resize_segmentation(r'D:\DeepRetinaSegmentationData\UBMI_test_data\Vessel_masks_labeled\nnUNet_predict_VesselsClass',
                        r'D:\DeepRetinaSegmentationData\UBMI_test_data\Vessel_masks_labeled\Fundus_Images',
                        r'D:\DeepRetinaSegmentationData\UBMI_test_data\Vessel_masks_labeled\nnUNet_predict_VesselsClass_orig')
    