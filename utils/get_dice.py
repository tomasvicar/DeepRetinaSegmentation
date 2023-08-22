import numpy as np


def get_dice(results,target):
    
    X = results.detach().cpu().numpy()>0
    Y = target.detach().cpu().numpy()>0
    

    TP = np.sum(((X==1)&(Y==1)).astype(np.float64))
    FP = np.sum(((X==1)&(Y==0)).astype(np.float64))
    FN = np.sum(((X==0)&(Y==1)).astype(np.float64))
    
    dice = (2 * TP )/ ((2 * TP) + FP + FN)
    
    
    return dice
    
    
    
    
    
    
    