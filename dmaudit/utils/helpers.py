import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import numpy as np


def get_criterion(criterion_name):
    if criterion_name == 'ce':
        return F.cross_entropy
    elif criterion_name == 'bce':
        return F.binary_cross_entropy_with_logits
    elif criterion_name == 'mse':
        return F.mse_loss
    else:
        raise NotImplementedError(f"Criterion {criterion_name} not found.")
    

def pretty_status_printer(architecture, 
                          percentages = None, 
                          artifact = None):
    print("----------------------------")
    print(architecture, f"artifact: {artifact}" if artifact is not None else "")
    if percentages is not None:
         print("Train/Val split of train folds:", percentages)
    print("----------------------------")


def  get_balance_subset_sampler(curr_subset, generator):
    controlling_dataset = curr_subset.dataset
    indices = np.array(curr_subset.indices)
    print('indices',indices)
    print('class counts',controlling_dataset.get_curr_class_counts(indices))
    reciprocal_weights = torch.tensor((1 / controlling_dataset.get_curr_class_counts(indices)).values)
    print('recip',reciprocal_weights)
    targets = torch.tensor(controlling_dataset.get_targets(indices).values)
    print('targ',targets)
    sample_weights = reciprocal_weights[targets.type(torch.long)]
    sampler = WeightedRandomSampler(sample_weights,len(curr_subset),generator=generator)
    return sampler



                