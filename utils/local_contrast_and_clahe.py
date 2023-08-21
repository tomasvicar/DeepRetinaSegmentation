import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import sobel
import cv2
from scipy.ndimage import binary_erosion
from skimage.morphology import disk
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.color import rgb2hsv, hsv2rgb


def local_contrast_and_clahe(img, fov, pix_per_deg):

    ClipLimit = 0.005
    # original value chosen for 55 pix_per_deg
    sigma = 80 / (55 / pix_per_deg)
    kernel = 250 / (55 / pix_per_deg)
    
    img = img.astype(np.float32) / 255 
    

    mask = binary_erosion(fov, disk(1))

    dt = distance_transform_edt(mask == 0)

    sx = sobel(dt, axis=0) / 4
    sy = sobel(dt, axis=1) / 4

    xx, yy = np.meshgrid(np.arange(dt.shape[1]), np.arange(dt.shape[0]))

    xxx = xx - sy * dt
    yyy = yy - sx * dt

    img_interp = cv2.remap(img, xxx.astype(np.float32), yyy.astype(np.float32), cv2.INTER_LINEAR)

    G = gaussian(img_interp, sigma, channel_axis = 2)
    img_interp = (img_interp - G) / G + 0.5
    img_interp[img_interp < 0] = 0
    img_interp[img_interp > 1] = 1

    img_interp = rgb2hsv(img_interp)
    img_interp[:,:,2] = equalize_adapthist(img_interp[:,:,2], kernel_size=kernel, clip_limit=ClipLimit)
    img_interp = hsv2rgb(img_interp)

    img_interp[fov == 0] = 0
    
    img_interp = img_interp * 255 
    img_interp[img_interp < 0] = 0
    img_interp[img_interp > 255] = 255
    img_interp = img_interp.astype(np.uint8) 

    return img_interp



if __name__ == "__main__":
    
    import sys
    sys.path.append("..")
    from datasets_loaders.LoaderGeneric import LoaderGeneric
    
    img = imread(r"..\..\databases\ARIA\aria_a_markups\aria_a_markups\aria_a_5_9.tif")
    img_norm = local_contrast_and_clahe(img, LoaderGeneric.get_fov_auto(img), 50)
    
    plt.imshow(img)
    plt.show()
    
    plt.imshow(img_norm)
    plt.show()