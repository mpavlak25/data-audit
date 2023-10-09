import numpy as np
import torch
import torch.nn.functional as F

from tamid.utils import log_data_to_csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')


def train(network, train_loader, criterion, optimizer, scheduler, epoch, log_path=None, log_interval=10, dry_run=False):
    network.train()
    network = network.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data.to(device))
        if criterion == F.mse_loss:
            target = target.float()
            output = output.squeeze()
        else:
            target = target.long()
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if dry_run:
            break

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        if log_path is not None:
            log = dict(
                epoch=epoch,
                progress=100. * batch_idx / len(train_loader),
                loss=loss.detach().item()
            )
            log_data_to_csv(log_path, log)
