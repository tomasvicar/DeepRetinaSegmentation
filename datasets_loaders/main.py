

from LoaderAria import LoaderAria
from LoaderAvrdb import LoaderAvrdb
from LoaderDrhagis import LoaderDrhagis
from LoaderDrishtigs import LoaderDrishtigs
from LoaderDrive import LoaderDrive
from LoaderG1020 import LoaderG1020
from LoaderHrf import LoaderHrf
from LoaderChasedb1 import LoaderChasedb1
from LoaderIostar import LoaderIostar
from LoaderOnhsd import LoaderOnhsd
from LoaderRefuge import LoaderRefuge
from LoaderRiga import LoaderRiga
from LoaderStare import LoaderStare
from LoaderUoa_dr import LoaderUoa_dr

import os

import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe

if __name__ == "__main__":
    
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '_normalized.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loaders_classes = [
        LoaderAria,
        LoaderAvrdb,
        LoaderDrhagis,
        LoaderDrishtigs,
        LoaderDrive,
        LoaderG1020,
        LoaderHrf,
        LoaderChasedb1,
        LoaderIostar,
        LoaderOnhsd,
        LoaderRefuge,
        LoaderRiga,
        LoaderStare,
        LoaderUoa_dr,
        ]
    
    for loaderclass in loaders_classes:
        print('______-----------------------------___________________')
        print('______-----------------------------___________________')
        print(loaderclass)
        print('______-----------------------------___________________')
        print('______-----------------------------___________________')
        loader = loaderclass(data_path, pix_per_deg, out_fname, preprocess_f)
        loader.preprocess()
        
        
        