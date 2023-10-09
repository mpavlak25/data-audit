# %%
import os
import warnings
from pathlib import Path
import glob
from dmaudit.configs.local import get_cfg_locals
from yacs.config import CfgNode as CN

# %%
# This file represents the configuration for experiment 1. We build off default and local configs, overriding where necessary 
# but only in such a way that this should be applicable regardless of machine.

# experiment override node
cfg = CN()
cfg.TRAIN = CN()
cfg.TRAIN.BINARY_TARGET = True
cfg.TRAIN.RUN_ARTIFACT = True
cfg.TRAIN.RUN_Y_PREDICTION = True


cfg.EXPERIMENT = CN()
cfg.EXPERIMENT.NAME = "Experiment_1_Scaling"


local_cfg = get_cfg_locals()
base_path = local_cfg.SYSTEM.BASEPATH
base_data_path = local_cfg.SYSTEM.PROCESSED_DATA_PATH

# Full path to dataframe for experiment runs. Contains pregenerated biased datasets for easy analysis
cfg.DATASET = CN()
cfg.DATASET.RUN_DFS = CN()
cfg.DATASET.RUN_DFS.LABEL_COLUMNS = ['artifact','is_mel']
cfg.DATASET.RUN_DFS.FOLDER_PATH = os.path.join(base_path, 'experiments/Experiment_1_Scaling/Experiment_1_Train_DFs')


subset_filenames = glob.glob(os.path.join(cfg.DATASET.RUN_DFS.FOLDER_PATH,'*'))
subset_filenames = [item for item in subset_filenames if 'fixedn_50_' in item ]
cfg.DATASET.RUN_DFS.FILENAMES = subset_filenames
assert len(cfg.DATASET.RUN_DFS.FILENAMES) > 0, "No df files found, please correct folder or base path"

#   must override in experiment, list of tuple of paths to datasets that will be used for 
#   each experiment respectively

NAT_DATA_PATH = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_NATURAL_EXTENSION)
noise_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'noise')
dark_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'dark')
compression_20_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_compression_quality_20')
compression_40_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_compression_quality_40')
compression_60_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_compression_quality_60')
compression_80_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_compression_quality_80')

cfg.DATASET.ARTIFICIAL_SETS_TO_RUN = [(NAT_DATA_PATH,noise_path),
                                     (NAT_DATA_PATH,dark_path),
                                     (NAT_DATA_PATH,compression_20_path),
                                     (NAT_DATA_PATH,compression_40_path),
                                     (NAT_DATA_PATH,compression_60_path),
                                     (NAT_DATA_PATH,compression_80_path)
                                     ]


cfg.MODEL = CN()
cfg.MODEL.ARCH_NAMES = ['resnet18']
cfg.MODEL.PRETRAINED = True
cfg.OUTPUT = CN()
cfg.OUTPUT.SAVE_PATH = os.path.join(base_path,'experiments/Experiment_1_Scaling/Experiment_1_Results')



# %%
