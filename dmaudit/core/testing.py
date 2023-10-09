import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from ibar.utils import log_data_to_csv
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')


def test(network, test_loader, label_str='dx', image_key='image_id', log_path=None, dry_run=False, compute_auc=True,
         return_preds=False, criterion=F.cross_entropy):
    network.eval()
    network = network.to(device)
    
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_image_ids = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if len(batch) == 2:
                data, target = batch
            else:
                data, target, meta = batch
            
            output = network(data.to(device))
            pred = output.cpu().data.max(1, keepdim=True)[1]
            probs = torch.softmax(output.cpu(), dim=-1)
            
            if criterion == F.mse_loss:
                target = target.float()
                output = output.squeeze()
            else:
                target = target.long()
            
            test_loss += criterion(output, target.to(device), reduction='sum').item()
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            
            if criterion == F.mse_loss:
                all_preds.append(output.cpu().squeeze())
            elif output.shape[-1] > 2:
                all_preds.append(pred.squeeze())
            else:
                all_preds.append(probs.cpu()[:, 1])
            all_labels.append(target.numpy())
            all_image_ids.append([m[image_key] for m in meta])
            if dry_run:
                break
    
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n')
    
    log = dict()
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_ids = sum(all_image_ids, [])
    if compute_auc and not dry_run:
        auc = roc_auc_score(all_labels, all_preds)
        print(f'AUC score: {auc:.4f}')
        log['auc'] = auc
    
    log['loss'] = test_loss
    log['acc'] = acc
    if log_path is not None:
        log_data_to_csv(log_path, log)
    
    if return_preds:
        df = pd.DataFrame()
        df['pred'] = all_preds
        df[label_str] = all_labels
        df[image_key] = all_ids
        return acc, log, df
    return acc, log
