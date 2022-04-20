import pydicom as dcm
import matplotlib.pyplot as plt

pathDCM = '/data/rj21/test/Data/A001_/series0022-Body/img0001-7.66418.dcm'

dataset = dcm.dcmread(pathDCM)

img = dataset.pixel_array

plt.Figure()
plt.imshow(img, cmap=plt.cm.gray)
