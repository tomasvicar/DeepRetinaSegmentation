import numpy as np 
import matplotlib.pyplot as plt
import cv2


def visboundaries(img,color='r',linewidth=2):


    img=(img>0).astype(np.uint8)*255
    
    
    contours, _= cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    for cnt in contours: 
      
        tmp1=np.concatenate((cnt[:,0,0],cnt[[0],0,0]))
        tmp2=np.concatenate((cnt[:,0,1],cnt[[0],0,1]))
    
        plt.plot(tmp1,tmp2,color=color,linewidth=linewidth)