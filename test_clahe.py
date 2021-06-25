import cv2
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt


data = "D:\DeepRetinaSegmentation\data_preprocessed\Images\drive_test_dr_03.png"


img = imread(data)


img=img.astype(np.float64)
img = (img/255)-0.5




img = img[:,:,1]
img = np.expand_dims(img,2)

# x = 384
# x = 128*4
x = 128
out_size=[x,x]

r1=200
r2=200
r=[r1,r2]

img=img[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1],:]


        


img = (img+0.5)*255
img[img<0] = 0
img[img>255] = 255
img=img.astype(np.uint8)

clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(4,4))
img = clahe.apply(img[:,:,0])

img = img.astype(np.float64)/255-0.5
img = np.expand_dims(img,2)


img = img[:128,:128,:]

plt.imshow(img[:,:,0]+0.5,vmin=0,vmax=1)
plt.show()


