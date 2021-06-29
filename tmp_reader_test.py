import time
import numpy as np
from skimage.io import imread
from PIL import Image



# start = time.time()


# xx = []
# for k in range(10):
#     mm = Image.open(r"D:\DeepRetinaSegmentation\data_preprocessed\Images\aria_na_amd_10027.png")
#     xx.append(mm)
#     end = time.time()
#     print(end - start)

    
for k in range(50):
    mmm = mm.crop([0,0,50,50])
    array = np.array(mmm)

end = time.time()
print(end - start)



# start = time.time()

# for k in range(50):
    
#     mm = imread(r"D:\DeepRetinaSegmentation\data_preprocessed\Images\aria_na_amd_10027.png")
    
#     array = mm[0:50,0:50]


# end = time.time()
# print(end - start)
