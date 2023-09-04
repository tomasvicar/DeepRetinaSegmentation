import numpy as np


def get_dice(results,target):
    
    X = results.detach().cpu().numpy()>0
    Y = target.detach().cpu().numpy()>0
    

    TP = np.sum(((X==1)&(Y==1)).astype(np.float64))
    FP = np.sum(((X==1)&(Y==0)).astype(np.float64))
    FN = np.sum(((X==0)&(Y==1)).astype(np.float64))
    
    dice = (2 * TP )/ ((2 * TP) + FP + FN)
    
    if np.isnan(dice):
        print('dice nan !!!!')
        return 0
        
    
    return dice


def get_dice_mask_type(results, target, mask_types, mask_type_use):
    
    
    Xs = results.detach().cpu().numpy()>0
    Ys = target.detach().cpu().numpy()>0
    

    TPs = 0
    FPs = 0
    FNs = 0
    
    for batch_idx, mask_type in enumerate(mask_types):
        mask_idx = mask_type.index(mask_type)


        X = Xs[batch_idx, mask_idx, ...]
        Y = Ys[batch_idx, 0, ...]

        TP = np.sum(((X==1)&(Y==1)).astype(np.float64))
        FP = np.sum(((X==1)&(Y==0)).astype(np.float64))
        FN = np.sum(((X==0)&(Y==1)).astype(np.float64))
        
        TPs = TPs + TP 
        FPs = FPs + FP 
        FNs = FNs + FN 
    
    dice = (2 * TPs )/ ((2 * TPs) + FPs + FNs)
    
    
    return dice
    
    
    
    
    
    
    