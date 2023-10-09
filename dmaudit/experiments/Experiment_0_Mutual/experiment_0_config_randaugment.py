# %%
import os
import warnings
from pathlib import Path
import glob
from dmaudit.configs.local import get_cfg_locals
from yacs.config import CfgNode as CN

# %%
# This file represents the configuration for experiment 0. We build off default and local configs, overriding where necessary 
# but only in such a way that this should be applicable regardless of machine.

## The purpose of experiment zero is to explore how task models are influenced by the introduction of biased 
# artifacts. These are artifacts we know we introduce at varying levels of relationship with label Y, and we 
# want to see how they are exploited by the model.


# experiment override node
cfg = CN()
cfg.TRAIN = CN()
cfg.TRAIN.BINARY_TARGET = True
cfg.TRAIN.RUN_ARTIFACT = False
cfg.TRAIN.RUN_Y_PREDICTION = True
cfg.TRAIN.BATCH_SIZE = 128
cfg.TEST = CN()
cfg.TEST.BATCH_SIZE = 128


cfg.EXPERIMENT = CN()
cfg.EXPERIMENT.NAME = "Experiment_0_Mutual"

# get local settings so we can make paths depend on the base paths and data paths 
local_cfg = get_cfg_locals()
base_path = local_cfg.SYSTEM.BASEPATH
base_data_path = local_cfg.SYSTEM.PROCESSED_DATA_PATH

cfg.DATASET = CN()
cfg.DATASET.AUGMENTATION = 'RandAugment'

cfg.DATASET.RUN_DFS = CN()
# predict this column
cfg.DATASET.RUN_DFS.Y_LABEL_COLUMN = 'is_malignant'

# Full path to dataframes for runs. Contains pregenerated biased datasets for easy analysis
cfg.DATASET.RUN_DFS.FOLDER_PATH = os.path.join(base_path, 'experiments/Experiment_0_Mutual/Experiment_0_Train_DFs')


subset_filenames = glob.glob(os.path.join(cfg.DATASET.RUN_DFS.FOLDER_PATH,'*'))
# fill in if you want to automatically generate a subset, otherwise runs on all dataframes
subset_filenames = [item for item in subset_filenames if '9.00' in item]

cfg.DATASET.RUN_DFS.FILENAMES = subset_filenames
assert len(cfg.DATASET.RUN_DFS.FILENAMES) > 0, "No df files found, please correct folder or base path"

#   must override in experiment, list of tuple of paths to datasets that will be used for 
#   each experiment respectively

NAT_DATA_PATH = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_NATURAL_EXTENSION)

compression_30_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_compression_quality_30')

# the only dataset we run on is the case of visually apparent compression
cfg.DATASET.ARTIFICIAL_SETS_TO_RUN = [(NAT_DATA_PATH, compression_30_path)]


cfg.MODEL = CN()

cfg.MODEL.ARCH_NAMES = ['swin_t']
cfg.MODEL.PRETRAINED = True
cfg.OUTPUT = CN()
cfg.OUTPUT.SAVE_PATH = os.path.join(base_path,'experiments/Experiment_0_Mutual/Experiment_0_RandAugment_Results')



# %%
