# %%
import os
import json
import torch
import argparse
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time

from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from dmaudit.models import get_model

from dmaudit.utils  import log_data_to_csv, AverageMeter, get_criterion, pretty_status_printer, get_balance_subset_sampler
from dmaudit.dataset import CsvDataset
from dmaudit.constants import IMNET_TRAIN_TRANSFORM, IMNET_W_RANDAUGMENT_TRAIN_TRANSFORM, IMNET_TEST_TRANSFORM

import click
from dmaudit.configs.defaults import get_cfg_defaults, combine_cfgs


# %%

@click.command()
@click.option('--experiment_cfg_path', type=click.Path(exists=True),
              help="CFG file containing experiment node which will be used to overwrite default behaviour.")
def combine_cfg_and_run(experiment_cfg_path):
    cfg = combine_cfgs(experiment_cfg_path)
    main(cfg)


def train(network,
          train_loader,
          criterion,
          optimizer,
          scheduler,
          epoch,
          data_load_meter=None,
          batch_time_meter=None,
          log_path=None,
          log_interval=10,
          dry_run=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.train()
    network = network.to(device)
    
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if data_load_meter is not None:
            data_load_meter.update(time.time() - end)
        optimizer.zero_grad()
        
        output = network(data.to(device))
        if criterion == F.mse_loss:
            target = target.float()
            output = output.squeeze()
        else:
            target = target.type(torch.LongTensor)
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
        if batch_time_meter is not None:
            batch_time_meter.update(time.time() - end)
        end = time.time()


def test(network, test_loader, label_str='artifact', image_key='image_id', log_path=None, dry_run=False, compute_auc=True,
         return_preds=False, criterion=F.cross_entropy,print_name = ''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.eval()
    network = network.to(device)
    
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_ids = []
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
                target = target.type(torch.LongTensor)
            
            test_loss += criterion(output, target.to(device), reduction='sum').item()
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            
            if criterion == F.mse_loss:
                all_preds.append(output.cpu().squeeze())
            elif output.shape[-1] > 2:
                all_preds.append(pred.squeeze().flatten())
            else:
                all_preds.append(probs.cpu()[:, 1].flatten())
            all_labels.append(target.numpy())
            if len(batch) == 3:
                all_ids.extend(meta[image_key])
            if dry_run:
                break
    
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    log = dict()
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if compute_auc and not dry_run:
        try: 
            auc = roc_auc_score(all_labels, all_preds)
            print(f'AUC score: {auc:.4f}')
            log['auc'] = auc
            print(f'\n{print_name} set: Avg. loss: {test_loss:.4f}, AUC: {auc}')
        except ValueError:
            print(f'\n{print_name} set: Avg. loss: {test_loss:.4f}. No positive examples, cannot calculate AUC.')
    else:
        print(f'\n{print_name} set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n')
    
    log['loss'] = test_loss
    log['acc'] = acc
    if log_path is not None:
        log_data_to_csv(log_path, log)
    
    if return_preds:
        df = pd.DataFrame()
        df['pred'] = all_preds
        # df[label_str] = all_labels
        df[image_key] = all_ids
        return acc, log, df
    return acc, log


def get_output_pathmap(cfg):
    """
    Function that based on the run's specified dataframe filenames gets a map of
    output paths that give an output save path for each dataframe path.
    """
    if cfg.OUTPUT.COMBINE_BY_DF:
        # based on all input dfs get output folders
        df_paths = cfg.DATASET.RUN_DFS.FILENAMES
        
        def path_to_folder(filepath):
            filename = os.path.split(filepath)[-1]
            return os.path.splitext(filename)[0].replace('.',',')
            
        
        output_path_map = {path: os.path.join(cfg.OUTPUT.SAVE_PATH, path_to_folder(path)) for path in df_paths}
        # create the paths
        for folder_path in output_path_map.values():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        return output_path_map
    else:
        raise NotImplementedError("Output is currently specified by dataframe")


def train_test_loop(cfg,
                    curr_output_path,
                    data_frame_path,
                    data_path_tuple, 
                    arch,
                    curr_label_column,
                    synthetic_artifact_name,
                    criterion,
                    epoch_time,
                    data_time,
                    batch_time,
                    generator,
                    test_results = True,
                    counterfactual_test = False
                    ):

    # Create temporary dataframes to store fold evaluation results before join
    if test_results:
        temp_test_results_df = pd.DataFrame()
    if counterfactual_test:
        temp_counterfactual_natural_df = pd.DataFrame()
        temp_counterfactual_artifact_df = pd.DataFrame()

    for k in range(cfg.DATASET.K_FOLDS):
        print(f"--- FOLD {k} ---")
        
        fold_output_path = os.path.join(curr_output_path,str(k))
        if not os.path.exists(fold_output_path):
            os.makedirs(fold_output_path)
        train_log = os.path.join(fold_output_path,'train.log')
        val_log = os.path.join(fold_output_path,'val.log')
        test_log = os.path.join(fold_output_path,'test.log')

        # Load models for this run
        model = get_model(arch, 
                        cfg.MODEL.NUM_LAST_LAYER_OUTPUTS,
                        pretrained=cfg.MODEL.PRETRAINED)                    
        if cfg.MODEL.WEIGHTS_PATH is not None:
            model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS_PATH, map_location='cpu'))
        
        # ----------------------------
        # Initialize Datasets and Dataloaders
        # ----------------------------
        print(IMNET_TRAIN_TRANSFORM if cfg.DATASET.AUGMENTATION.lower() == 'basic' else IMNET_W_RANDAUGMENT_TRAIN_TRANSFORM)
        trainval_ds = CsvDataset(metadata_path = data_frame_path,
                                test_folds = [k],
                                split = 'train',
                                paths = data_path_tuple,
                                select_path_col = cfg.DATASET.RUN_DFS.SELECT_IMAGES_PATH_COLUMN,
                                label_col = curr_label_column,
                                transform = IMNET_TRAIN_TRANSFORM if cfg.DATASET.AUGMENTATION == 'basic' else IMNET_W_RANDAUGMENT_TRAIN_TRANSFORM)
        
        test_ds = CsvDataset(metadata_path = data_frame_path,
                                test_folds = [k],
                                split = 'test',
                                return_meta=True,
                                paths = data_path_tuple,
                                select_path_col = cfg.DATASET.RUN_DFS.SELECT_IMAGES_PATH_COLUMN,
                                label_col = curr_label_column,
                                transform = IMNET_TEST_TRANSFORM)
        
        len_train = int(cfg.DATASET.TRAIN_VAL_PERCENTAGES[0]*len(trainval_ds))
        len_val = len(trainval_ds) - len_train
        train_ds, val_ds = torch.utils.data.random_split(trainval_ds,
                                                        [len_train,len_val],
                                                        generator=torch.Generator().manual_seed(cfg.SYSTEM.SEED))
        print(f"len train set {len(train_ds)}, len val set {len(val_ds)}")
        
        if cfg.TRAIN.BALANCE_CLASSES:
            BalancedRandomSampler = get_balance_subset_sampler(train_ds,generator=generator)
            train_loader = DataLoader(train_ds,
                                    batch_size = cfg.TRAIN.BATCH_SIZE,
                                    shuffle = False,
                                    sampler=BalancedRandomSampler,
                                    num_workers = cfg.SYSTEM.NUM_WORKERS,
                                    pin_memory=True)
        
        else:
            train_loader = DataLoader(train_ds,
                                    batch_size = cfg.TRAIN.BATCH_SIZE,
                                    shuffle = True,
                                    num_workers = cfg.SYSTEM.NUM_WORKERS,
                                    pin_memory=True)
        
        val_loader = DataLoader(val_ds,
                                batch_size = cfg.TRAIN.BATCH_SIZE)

        if not cfg.TRAIN.EVAL_ONLY:
            # Store argument details. TODO: Make this specific to the DF
            with open(os.path.join(fold_output_path,'args.yaml'),'w') as f:
                f.write(str(cfg))
            
            if cfg.TRAIN.OPTIMIZER == 'SGD':
                optimizer = optim.SGD(model.parameters(), 
                                    lr = cfg.TRAIN.LR, 
                                    weight_decay = cfg.TRAIN.WEIGHT_DECAY, 
                                    momentum = cfg.TRAIN.MOMENTUM[0])
            elif cfg.TRAIN.OPTIMIZER == 'AdamW':
                optimizer = optim.AdamW(model.parameters(),
                                        lr = cfg.TRAIN.LR, 
                                        weight_decay = cfg.TRAIN.WEIGHT_DECAY, 
                                        betas = cfg.TRAIN.MOMENTUM)
            else:
                raise NotImplementedError(f"Optimizer {cfg.TRAIN.OPTIMIZER} not implemented.")
            if cfg.TRAIN.SCHEDULER == 'LINEAR_LR_DECAY':
                scheduler = StepLR(optimizer, 
                                step_size = len(train_ds), 
                                gamma = cfg.TRAIN.SCHEDULER_GAMMA)
            else:
                raise NotImplementedError("Only linear decay implemented.")

            best_acc = 0
            best_loss = 1e8
            end_time = time.time()
            for epoch in range(1, cfg.TRAIN.NUM_EPOCHS + 1):
                # ------- train for one epoch -------
                train(model, train_loader, criterion, optimizer, scheduler, 
                      epoch, log_path = train_log, data_load_meter=data_time,
                      batch_time_meter=batch_time, dry_run = cfg.TRAIN.DO_DRY_RUN)
                
                epoch_time.update(time.time()-end_time)
                
                print(epoch_time)
                print(data_time)
                print(batch_time)

                end_time = time.time()
                # ------- perform evaluation -------
                acc, out = test(model, 
                                val_loader, 
                                log_path = val_log, 
                                dry_run = cfg.TRAIN.DO_DRY_RUN,
                                compute_auc = cfg.TRAIN.BINARY_TARGET,
                                criterion = criterion,
                                print_name="VAL")
                
                # ------- save if necessary -------
                ##### ADJUST TO SAVE BY FOLD
                if not cfg.TRAIN.DO_DRY_RUN:
                    if criterion == F.mse_loss and out['loss'] < best_loss:
                        best_loss = out['loss']
                        print(f'New best loss: {best_loss:.3f}')
                        torch.save(model.state_dict(), os.path.join(fold_output_path, 'best.pth'))
                    elif acc > best_acc:
                        print(f'New best acc: {acc:.3f}')
                        best_acc = acc
                        torch.save(model.state_dict(), os.path.join(fold_output_path, 'best.pth'))
                else:
                    break

            print(f'{synthetic_artifact_name} Fold {k} Complete. Testing...')

            ## ------- Test this fold's best model -------
            if test_results or counterfactual_test:
                model.load_state_dict(torch.load(os.path.join(fold_output_path,'best.pth'), map_location='cpu'))
            if test_results:
                orig_df_test_loader = DataLoader(test_ds, batch_size = cfg.TEST.BATCH_SIZE)
                
                test_acc, log_out, test_df = test(model,
                                                orig_df_test_loader,
                                                label_str=curr_label_column,
                                                log_path=val_log,
                                                dry_run=cfg.TRAIN.DO_DRY_RUN,
                                                compute_auc=cfg.TRAIN.BINARY_TARGET,
                                                criterion=criterion,
                                                return_preds=True,
                                                print_name = "TEST")
                # print(f"test accuracy {test_acc}")

                test_df.rename(columns={'pred':f"pred-{curr_label_column}-{arch}-{synthetic_artifact_name}-orig_dset_ratio"},inplace=True)
                temp_test_results_df = pd.concat((temp_test_results_df,test_df))

            ## ------- Run counterfactual tests if desired -------
            if counterfactual_test:
                if cfg.TEST.COUNTERFACTUAL_INDEX is not None:
                    # force all natural images
                    test_ds.override_select_path_col = cfg.TEST.COUNTERFACTUAL_INDEX

                    natural_df_test_loader = DataLoader(test_ds, batch_size = cfg.TEST.BATCH_SIZE)

                    _, _, natural_df = test(model, 
                                            natural_df_test_loader,
                                            log_path=val_log,
                                            label_str=curr_label_column,
                                            dry_run=cfg.TRAIN.DO_DRY_RUN,
                                            compute_auc=cfg.TRAIN.BINARY_TARGET,
                                            criterion=criterion,
                                            return_preds=True,
                                            print_name = "TEST NATURAL")
                    #TODO: Add ratio
                    natural_df.rename(columns={'pred':f"pred-{curr_label_column}-{arch}-{synthetic_artifact_name}-counterfactual_natural_images"},inplace=True)
                    temp_counterfactual_natural_df = pd.concat((temp_counterfactual_natural_df,natural_df))
                else:    

                    # force all natural images
                    test_ds.override_select_path_col = 0

                    natural_df_test_loader = DataLoader(test_ds, batch_size = cfg.TEST.BATCH_SIZE)

                    _, _, natural_df = test(model, 
                                            natural_df_test_loader,
                                            log_path=val_log,
                                            label_str=curr_label_column,
                                            dry_run=cfg.TRAIN.DO_DRY_RUN,
                                            compute_auc=cfg.TRAIN.BINARY_TARGET,
                                            criterion=criterion,
                                            return_preds=True,
                                            print_name = "TEST NATURAL")
                    #TODO: Add ratio
                    natural_df.rename(columns={'pred':f"pred-{curr_label_column}-{arch}-{synthetic_artifact_name}-counterfactual_natural_images"},inplace=True)
                    temp_counterfactual_natural_df = pd.concat((temp_counterfactual_natural_df,natural_df))
                    
                    # force all artifact images
                    test_ds.override_select_path_col = 1

                    artifact_df_test_loader = DataLoader(test_ds, batch_size = cfg.TEST.BATCH_SIZE)

                    _, _,artifact_df = test(model,
                                            artifact_df_test_loader,
                                            log_path=val_log,
                                            label_str=curr_label_column,
                                            dry_run=cfg.TRAIN.DO_DRY_RUN,
                                            compute_auc=cfg.TRAIN.BINARY_TARGET,
                                            criterion=criterion,
                                            return_preds=True,
                                            print_name = "TEST ARTIFACT")                
                    
                    artifact_df.rename(columns={'pred':f"pred-{curr_label_column}-{arch}-{synthetic_artifact_name}-counterfactual_artifact_images"},inplace=True)
                    temp_counterfactual_artifact_df = pd.concat((temp_counterfactual_artifact_df,artifact_df))
                    
                    # revert to whichever artifacts the dataframe specified.
                test_ds.override_select_path_col = None
                
    if test_results and counterfactual_test:
        if cfg.TEST.COUNTERFACTUAL_INDEX:
            return temp_test_results_df, temp_counterfactual_natural_df
        else:
            return temp_test_results_df, temp_counterfactual_natural_df, temp_counterfactual_artifact_df
    elif test_results:
        return (temp_test_results_df,)


def main(cfg):
    # ------- Seeds -------
    torch.manual_seed(cfg.SYSTEM.SEED)
    np.random.seed(cfg.SYSTEM.SEED)
    random.seed(cfg.SYSTEM.SEED)
    # torch.backends.cudnn.benchmark = True
    
    # ------- Get paths -------
    save_path = cfg.OUTPUT.SAVE_PATH
    output_path_map = get_output_pathmap(cfg)
    
    print('Configuration\n', cfg)
    
    # ------- Create meters for profiling -------
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Load Time', ':6.3f')
    epoch_time = AverageMeter('Epoch Time', ':6.3f')
    eval_df = pd.DataFrame()
    
    # TODO: Update get_criterion to be a map that is specific to the run
    criterion = get_criterion(cfg.TRAIN.CRITERION)

    for data_frame_path, base_output_path in output_path_map.items():
        
        # ------- Overall results stores for df -------
        print(f"DF Ratio: {data_frame_path.split('/')[-1].split('_')[-1][:-4]}")
        to_export_results_df = pd.read_csv(data_frame_path)
        
        for data_path_tuple in cfg.DATASET.ARTIFICIAL_SETS_TO_RUN:
            for arch in cfg.MODEL.ARCH_NAMES:
                synthetic_artifact_name = os.path.split(data_path_tuple[1])[-1]
                
                pretty_status_printer(arch,artifact=synthetic_artifact_name,percentages=cfg.DATASET.TRAIN_VAL_PERCENTAGES)
                
                orig_ratio_eval_df = pd.DataFrame()
                curr_output_path = os.path.join(base_output_path, synthetic_artifact_name, arch)
                
                if not os.path.exists(curr_output_path):
                    os.makedirs(curr_output_path)

                # Train k folds of artifact prediction cfg.DATASET.RUN_DFS.LABEL_COLUMNS = ['artifact','is_mel']
                if cfg.TRAIN.RUN_ARTIFACT:
                    print("Train artifact")
                    results = train_test_loop(cfg,
                                              os.path.join(curr_output_path,'attribute_prediction'),
                                              data_frame_path,
                                              data_path_tuple,
                                              arch,
                                              cfg.DATASET.RUN_DFS.ARTIFACT_COLUMN,
                                              synthetic_artifact_name,
                                              criterion,
                                              epoch_time,
                                              data_time,
                                              batch_time,
                                              generator=torch.Generator().manual_seed(cfg.SYSTEM.SEED),
                                              test_results= True,
                                              counterfactual_test = False
                                              )
                    for res in results:
                        to_export_results_df = to_export_results_df.merge(res,on='image_id')

                if cfg.TRAIN.RUN_Y_PREDICTION:
                    print("Train dx")
                    results = train_test_loop(cfg,
                                              os.path.join(curr_output_path,'dx_prediction'),
                                              data_frame_path,
                                              data_path_tuple,
                                              arch,
                                              cfg.DATASET.RUN_DFS.Y_LABEL_COLUMN,
                                              synthetic_artifact_name,
                                              criterion,
                                              epoch_time,
                                              data_time,
                                              batch_time,
                                              generator=torch.Generator().manual_seed(cfg.SYSTEM.SEED),
                                              test_results= True,
                                              counterfactual_test = True
                                              )
                    for res in results:
                        to_export_results_df = to_export_results_df.merge(res,on='image_id')
        #we export results for each dataframe, meaning we do not need to include ratio
        to_export_results_df.to_csv(os.path.join(base_output_path,'export_all.csv'))
                    


if __name__ == '__main__':
    combine_cfg_and_run()

# %%
