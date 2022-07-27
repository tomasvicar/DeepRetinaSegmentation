import torch

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