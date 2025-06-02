import torch 
import torch.nn as nn 
from torch.optim import Optimizer 
from typing import Dict 

def prune_optimizer(optimizer: Optimizer, mask): 
    optimizable_tensors = {} 
    for group in optimizer.param_groups: 
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None: 
            stored_state['exp_avg'] = stored_state['exp_avg'][mask]
            stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

            del optimizer.state[group['params'][0]] 
            group['params'][0] = nn.Parameter(group['params'][0][mask], requires_grad=True) 
            optimizer.state[group['params'][0]] = stored_state 
        else: 
            group['params'][0] = nn.Parameter(group['params'][0][mask], requires_grad=True)  
        optimizable_tensors[group['name']] = group['params'][0] 
    return optimizable_tensors 

def add_tensors_to_optimizer(optimizer: Optimizer, tensors_dict: Dict): 
    optimizable_tensors = {} 
    for group in optimizer.param_groups: 
        stored_state = optimizer.state.get(group['params'][0], None) 
        new_tensor = tensors_dict[group['name']]
        if stored_state is not None: 
            stored_state['exp_avg'] = torch.cat([stored_state['exp_avg'], torch.zeros_like(new_tensor)], dim=0)
            stored_state['exp_avg_sq'] = torch.cat([stored_state['exp_avg_sq'], torch.zeros_like(new_tensor)], dim=0) 

            del optimizer.state[group['params'][0]] 
            group['params'][0] = nn.Parameter(torch.cat([group['params'][0], new_tensor], dim=0), requires_grad=True)
            optimizer.state[group['params'][0]] = stored_state 
        else: 
            group['params'][0] = nn.Parameter(torch.cat([group['params'][0], new_tensor], dim=0), requires_grad=True)
        optimizable_tensors[group['name']] = group['params'][0]
    return optimizable_tensors 

def replace_tensor_in_optimizer(optimizer: Optimizer, name, new_tensor):
    optimizable_tensors = {} 
    for group in optimizer.param_groups: 
        if group['name'] == name: 
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None: 
                stored_state['exp_avg'] = torch.zeros_like(group['params'][0]) 
                stored_state['exp_avg_sq'] = torch.zeros_like(group['params'][0]) 

                del optimizer.state[group['params'][0]] 
                group['params'][0] = nn.Parameter(new_tensor, requires_grad=True)
                optimizer.state[group['params'][0]] = stored_state
            else: 
                group['params'][0] = nn.Parameter(new_tensor, requires_grad=True) 
            optimizable_tensors[group['name']] = group['params'][0]
    return optimizable_tensors