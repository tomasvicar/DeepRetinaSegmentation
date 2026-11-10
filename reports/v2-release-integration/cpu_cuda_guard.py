import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
def blocked(*args, **kwargs):
    raise RuntimeError('Release integration CPU check forbids CUDA initialization')
torch.cuda.init = blocked
torch.cuda._lazy_init = blocked
