# %%
import os
import warnings
from pathlib import Path
import glob
from dmaudit.constants.ham import HAM_NUM_CATEGORIES
from dmaudit.configs.local import get_cfg_locals
from yacs.config import CfgNode as CN

# %%
# This file represents the configuration for experiment 5. We build off default and local configs, overriding where necessary 
# but only in such a way that this should be applicable regardless of machine.

## For this trial we have the case of an artifact with varying strength that is randomly distributed 
# among samples. Because we know there are not other features that can be used as the distribution is 
# randomized, we can be confident that AUC will correspond to how easy it is to detect the artifacts.

# experiment override node
curr_attribute = 'dark'
cfg = CN()
cfg.TRAIN = CN()
cfg.TRAIN.BINARY_TARGET = True

cfg.TRAIN.RUN_ARTIFACT = False
# here we do not care about diagnosis model performance. 
cfg.TRAIN.RUN_Y_PREDICTION = True

cfg.EXPERIMENT = CN()
cfg.EXPERIMENT.NAME = "Experiment_6_Multiple"
cfg.TEST = CN()
cfg.TEST.COUNTERFACTUAL_INDEX = -1

# get local settings so we can make paths depend on the base paths and data paths 
local_cfg = get_cfg_locals()
base_path = local_cfg.SYSTEM.BASEPATH
base_data_path = local_cfg.SYSTEM.PROCESSED_DATA_PATH

cfg.DATASET = CN()
cfg.DATASET.RUN_DFS = CN()
# predict this column
cfg.DATASET.AUGMENTATION = 'RandAugment'
cfg.DATASET.RUN_DFS.ARTIFACT_COLUMN = 'artifact'
cfg.DATASET.RUN_DFS.SELECT_IMAGES_PATH_COLUMN = 'artifact'
cfg.DATASET.RUN_DFS.Y_LABEL_COLUMN = 'is_malignant'

# Full path to dataframes for runs. Contains pregenerated biased datasets for easy analysis
cfg.DATASET.RUN_DFS.FOLDER_PATH = os.path.join(base_path, 'experiments/Experiment_6_Multiple/Experiment_6_Train_DFs')


subset_filenames = glob.glob(os.path.join(cfg.DATASET.RUN_DFS.FOLDER_PATH,'*'))

cfg.DATASET.RUN_DFS.FILENAMES = [f for f in subset_filenames if 'nclasses_4' in f]
print("-----------------------") 
print("RUNNING SUBSET:")
print(cfg.DATASET.RUN_DFS.FILENAMES)
print("-----------------------") 
assert len(cfg.DATASET.RUN_DFS.FILENAMES) > 0, "No df files found, please correct folder or base path"

#   must override in experiment, list of tuple of paths to datasets that will be used for 
#   each experiment respectively

NAT_DATA_PATH = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_NATURAL_EXTENSION)
dark_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'dark')
bright_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'bright')
red_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'color_patch_[0]')
green_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'color_patch_[1]')
# the only dataset we run on is the case of visually apparent compression
cfg.DATASET.ARTIFICIAL_SETS_TO_RUN = [(dark_path,bright_path,red_path,green_path,NAT_DATA_PATH)]

cfg.MODEL = CN()
# This is primarily attribute prediction, so we assume a strong model would do even better
cfg.MODEL.NUM_LAST_LAYER_OUTPUTS = 2
cfg.MODEL.ARCH_NAMES = ['swin_t']
cfg.MODEL.PRETRAINED = True
cfg.OUTPUT = CN()
cfg.OUTPUT.SAVE_PATH = os.path.join(base_path,'experiments/Experiment_6_Multiple/Experiment_6_4classes_DX')



# %%
