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

## The purpose of this trial is to demonstrate that models can learn to predict invisible, nonexistent artifacts with
# above random chance test performance if these artifacts have relationships with other groups in our dataset. 
# In our next experiment, we'll cover how this problem can be addressed. 


# experiment override node
cfg = CN()
cfg.TRAIN = CN()
cfg.TRAIN.BINARY_TARGET = True

cfg.TRAIN.RUN_ARTIFACT = True
# here we do not care about diagnosis model performance. 
cfg.TRAIN.RUN_Y_PREDICTION = False

cfg.EXPERIMENT = CN()
cfg.EXPERIMENT.NAME = "Experiment_1_Invisible"

# get local settings so we can make paths depend on the base paths and data paths 
local_cfg = get_cfg_locals()
base_path = local_cfg.SYSTEM.BASEPATH
base_data_path = local_cfg.SYSTEM.PROCESSED_DATA_PATH

cfg.DATASET = CN()
cfg.DATASET.AUGMENTATION = 'basic'
cfg.DATASET.RUN_DFS = CN()
# predict this column
cfg.DATASET.RUN_DFS.ARTIFACT_COLUMN = 'artifact'

# Full path to dataframes for runs. Contains pregenerated biased datasets for easy analysis
cfg.DATASET.RUN_DFS.FOLDER_PATH = os.path.join(base_path, 'experiments/Experiment_1_Invisible_Artifact/Experiment_1_Train_DFs')


subset_filenames = glob.glob(os.path.join(cfg.DATASET.RUN_DFS.FOLDER_PATH,'*'))
subset_filenames = [item for item in subset_filenames if '' in item ]
cfg.DATASET.RUN_DFS.FILENAMES = subset_filenames
assert len(cfg.DATASET.RUN_DFS.FILENAMES) > 0, "No df files found, please correct folder or base path"

#   must override in experiment, list of tuple of paths to datasets that will be used for 
#   each experiment respectively

NAT_DATA_PATH = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_NATURAL_EXTENSION)



# invisible artifact -- the artifact set is unmodified images
cfg.DATASET.ARTIFICIAL_SETS_TO_RUN = [(NAT_DATA_PATH, NAT_DATA_PATH)]


cfg.MODEL = CN()
# This is primarily attribute prediction, so we assume a strong model would do even better
cfg.MODEL.ARCH_NAMES = ['resnet18']
cfg.MODEL.PRETRAINED = True
cfg.OUTPUT = CN()
cfg.OUTPUT.SAVE_PATH = os.path.join(base_path,'experiments/Experiment_1_Invisible_Artifact/Experiment_1_Results')



# %%
