import torch
import torch.nn.functional as F

def l1_loss(result, target):
    
    cuda_check = result.is_cuda
    if cuda_check:
        cuda_device = result.get_device()
        device = torch.device('cuda:' + str(cuda_device) )
    target=target.to(device)
    
    return torch.mean(torch.abs(result - target))

def l2_loss(result, target):
    
    cuda_check = result.is_cuda
    if cuda_check:
        cuda_device = result.get_device()
        device = torch.device('cuda:' + str(cuda_device) )
    target=target.to(device)
    
    return torch.mean((result - target)**2)


def dice_loss_logit_mask_type(result, target, mask_types, mask_type_use):
  
    cuda_check = result.is_cuda
    if cuda_check:
        cuda_device = result.get_device()
        device = torch.device('cuda:' + str(cuda_device) )
    target=target.to(device)
    
  
    result=torch.sigmoid(result)
    
    intersection_sum = 0.
    A_sums = 0.
    B_sums = 0.
    
    
    for batch_idx, mask_type in enumerate(mask_types):
        mask_idx = mask_type.index(mask_type)
        
        
        result_tmp = result[batch_idx, mask_idx, ...]
        target_tmp = target[batch_idx, 0, ...]
        
        iflat = result_tmp.contiguous().view(-1)
        tflat = target_tmp.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
    
        A_sum = torch.sum(iflat)
        B_sum = torch.sum(tflat)
        
        
        intersection_sum = intersection_sum + intersection
        A_sums = A_sums + A_sum
        B_sums = B_sums + B_sum
    
    smooth = 1.
    
    return 1 - ((2. * intersection_sum + smooth) / (A_sums + B_sums + smooth) )





def dice_loss_logit(result, target):
  
    cuda_check = result.is_cuda
    if cuda_check:
        cuda_device = result.get_device()
        device = torch.device('cuda:' + str(cuda_device) )
    target=target.to(device)
    
  
    result=torch.sigmoid(result)
    smooth = 1.

    iflat = result.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )



def bce_logit(result, target):
    
    cuda_check = result.is_cuda
    if cuda_check:
        cuda_device = result.get_device()
        device = torch.device('cuda:' + str(cuda_device) )
    target=target.to(device)
    
  
    result=torch.sigmoid(result)
    
    return F.binary_cross_entropy(result,target)
    

    