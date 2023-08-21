
import numpy as np

def get_bbox(binary_img):
    
    rows = np.any(binary_img, axis=1)
    cols = np.any(binary_img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_to_bbox(img, bbox):
    
    rmin, rmax, cmin, cmax = bbox
    return img[rmin:rmax+1, cmin:cmax+1]



