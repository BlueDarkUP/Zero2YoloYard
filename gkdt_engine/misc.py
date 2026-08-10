import os
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
from torch import Tensor


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_local_rank():
    """Get local rank for current process"""
    if not is_dist_avail_and_initialized():
        return 0
    
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    else:
        return 0


def init_distributed_mode():
    # Check for torchrun/torch.distributed.launch (single-node or multi-node multi-GPU)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    # Single GPU mode
    else:
        print('Not using distributed mode - single GPU training')
        return False, 0, 1, 0
    
    # Distributed mode
    print(f'Distributed mode: rank {rank}/{world_size}, local_rank {local_rank}')
    
    # Set device before initializing process group to avoid NCCL warning
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Initialize process group with NCCL backend for GPU, GLOO for CPU
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()
    
    return True, rank, world_size, local_rank


# Store original print function
_builtin_print = print


def print_rank0(*args, **kwargs):
    """Print only on rank 0 process in distributed training"""
    if is_main_process():
        _builtin_print(*args, **kwargs)


def setup_print_for_distributed(is_distributed=True):
    """Replace built-in print with rank0-only print for distributed training.
    
    Args:
        is_distributed: If False, keeps normal print behavior
    """
    import builtins
    if is_distributed:
        builtins.print = print_rank0

def gather_data_from_all_gpus(tensor_data: torch.Tensor, cat_dim=0):
    """Note tensor_data should have the same size in all GPUs and be sent to cuda device before gathering!
    dim: which dimension to catch.
    """
    if tensor_data.is_cuda == False:  # ensure the tensor is in GPU
        tensor_data = tensor_data.cuda() 

    gathered = [torch.zeros_like(tensor_data) for _ in range(get_world_size())]  # create container

    # dist.barrier()  # Add barrier before critical operations. Ensure all tensors are created
    
    dist.all_gather(gathered, tensor_data)
    gathered = torch.cat(gathered, dim=cat_dim)
    
    return gathered

def gather_different_sized_tensors(tensor):
    """Gather tensors that may have different sizes at dim 0 across GPUs
    tensor: bs x (...), the varying size of dim can only be at dim 0!
    """
    if tensor.is_cuda == False:  # ensure the tensor is in GPU
        tensor = tensor.cuda() 

    world_size = dist.get_world_size()
    
    # Get local tensor size
    local_size = torch.tensor(tensor.size(0), device=tensor.device, dtype=torch.long)
    sizes = [torch.tensor(0, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes)
    
    # Pad local tensor to max size
    if local_size < max_size:
        padding = torch.zeros(max_size - local_size, *tensor.shape[1:], 
                            device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding])
    
    # Gather all padded tensors
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    # Remove padding
    gathered = [t[:sizes[i]] for i, t in enumerate(gathered)]
    return torch.cat(gathered)




if __name__ == '__main__':
    pass
    # =========backup below codes for gathering data=======
    # if misc.is_dist_avail_and_initialized():  # Gather results from all GPUs during distributed computation!
    #     acc_against_threshs = torch.tensor(pck_metric.acc_list, dtype=torch.float64).permute(1, 0).cuda()  # num_episodes x num_threshs 
    #     acc_all_gathered = misc.gather_different_sized_tensors(acc_against_threshs)  # (nGPUs*num_episodes) x num_threshs
    #     acc_mean, interval = pck_metric.get_mean_accuracy_result(acc_all_gathered.permute(1, 0).cpu().numpy().tolist())
    #     ne_data = torch.tensor(ne_metric.ne_list).cuda()  # num_episodes
    #     ne_all_gathered = misc.gather_data_from_all_gpus(ne_data)  # (nGPUs*num_episodes)
    #     ne_mean, ne_interval = ne_metric.get_mean_ne_result(ne_all_gathered.cpu().numpy().tolist())
    #     print('Gathered (all GPUs): Acc {}, Int. {}, NE {:.6f}, Int. {:.6f}'.format(acc_mean, interval, ne_mean, ne_interval))   
    #======================================================
