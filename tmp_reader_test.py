import time
import numpy as np
from skimage.io import imread
from PIL import Image
import h5py


# start = time.time()


# xx = []
# for k in range(10):
#     mm = Image.open(r"D:\DeepRetinaSegmentation\data_preprocessed\Images\aria_na_amd_10027.png")
#     xx.append(mm)
#     end = time.time()
#     print(end - start)

    
# for k in range(50):
#     mm = Image.open(r"D:\DeepRetinaSegmentation\data_preprocessed\Images\aria_na_amd_10027.png")
#     mmm = mm.crop([0,0,50,60])
#     array = np.array(mmm)

# end = time.time()
# print(end - start)


# mm = imread(r"D:\DeepRetinaSegmentation\data_preprocessed\Images\aria_na_amd_10027.png")

# shape = mm.shape
# with h5py.File('../tmp.hdf5',"w") as f:
    
#     start = time.time()
#     for k in range(300):
#         if k%100 ==0:
#             print(k)
#             end = time.time()
#             print(end - start)
#             start = time.time()
#         dset_img = f.create_dataset('aria_na_amd_10027' + str(k), (shape[0],shape[1],3), dtype='u1',chunks=(100,100,3),compression="gzip",compression_opts=4)
#         dset_img[:,:,:] = mm


# start = time.time()


# with h5py.File('../tmp.hdf5',"r") as h5dat_tmp:
            
#     for k in range(300):
        
#         array = h5dat_tmp["aria_na_amd_10027" + str(k)][80:130,80:130,:]
    
    

# end = time.time()
# print(end - start)







# start = time.time()

# for k in range(300):
    
#     mm = imread(r"D:\DeepRetinaSegmentation\data_preprocessed\Images\aria_na_amd_10027.png")
    
#     array = mm[0:50,0:50,:]

# end = time.time()
# print(end - start)






# with h5py.File(r"D:\DeepRetinaSegmentation\data_preprocessed_hdf5\dataset.hdf5","r") as f:
    
#     f


