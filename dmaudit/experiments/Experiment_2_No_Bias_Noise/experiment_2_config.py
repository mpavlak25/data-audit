# %%
import os
import warnings
from pathlib import Path
import glob
from dmaudit.configs.local import get_cfg_locals
from yacs.config import CfgNode as CN

# %%
# This file represents the configuration for experiment 2. We build off default and local configs, overriding where necessary 
# but only in such a way that this should be applicable regardless of machine.

## For this trial we have the case of an artifact with varying strength that is randomly distributed 
# among samples. Because we know there are not other features that can be used as the distribution is 
# randomized, we can be confident that AUC will correspond to how easy it is to detect the artifacts.

# experiment override node
cfg = CN()
cfg.TRAIN = CN()
cfg.TRAIN.BINARY_TARGET = True

cfg.TRAIN.RUN_ARTIFACT = True
# here we do not care about diagnosis model performance. 
cfg.TRAIN.RUN_Y_PREDICTION = False

cfg.EXPERIMENT = CN()
cfg.EXPERIMENT.NAME = "Experiment_2_No_Bias_Artifact_Noise"

# get local settings so we can make paths depend on the base paths and data paths 
local_cfg = get_cfg_locals()
base_path = local_cfg.SYSTEM.BASEPATH
base_data_path = local_cfg.SYSTEM.PROCESSED_DATA_PATH

cfg.DATASET = CN()
cfg.DATASET.RUN_DFS = CN()
# predict this column
cfg.DATASET.AUGMENTATION = 'basic'
cfg.DATASET.RUN_DFS.ARTIFACT_COLUMN = 'artifact'

# Full path to dataframes for runs. Contains pregenerated biased datasets for easy analysis
cfg.DATASET.RUN_DFS.FOLDER_PATH = os.path.join(base_path, 'experiments/Experiment_2_No_Bias_Noise/Experiment_2_Train_DFs')


subset_filenames = glob.glob(os.path.join(cfg.DATASET.RUN_DFS.FOLDER_PATH,'*'))
subset_filenames = [item for item in subset_filenames if '' in item ]
cfg.DATASET.RUN_DFS.FILENAMES = subset_filenames
assert len(cfg.DATASET.RUN_DFS.FILENAMES) > 0, "No df files found, please correct folder or base path"

#   must override in experiment, list of tuple of paths to datasets that will be used for 
#   each experiment respectively

NAT_DATA_PATH = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_NATURAL_EXTENSION)
noise_level_0_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0')
noise_level_001_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0.001')
noise_level_01_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0.01')
noise_level_05_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0.05')
noise_level_1_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0.1')
noise_level_2_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0.2')
noise_level_3_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0.3')
noise_level_4_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0.4')
noise_level_5_path = os.path.join(base_data_path, local_cfg.SYSTEM.HAM_ARTIFACT_EXTENSION,'global_noise_0.5')

# the only dataset we run on is the case of visually apparent compression
cfg.DATASET.ARTIFICIAL_SETS_TO_RUN = [(NAT_DATA_PATH, noise_level_0_path),
                                      (NAT_DATA_PATH, noise_level_001_path),
                                      (NAT_DATA_PATH, noise_level_01_path),
                                      (NAT_DATA_PATH, noise_level_05_path),
                                      (NAT_DATA_PATH, noise_level_1_path),
                                      (NAT_DATA_PATH, noise_level_2_path),
                                      (NAT_DATA_PATH, noise_level_3_path),
                                      (NAT_DATA_PATH, noise_level_4_path),
                                      (NAT_DATA_PATH, noise_level_5_path),
                                    #   (NAT_DATA_PATH, compression_40_path),
                                    #   (NAT_DATA_PATH, compression_50_path),
                                    #   (NAT_DATA_PATH, compression_60_path),
                                      
                                      ]

cfg.MODEL = CN()
# This is primarily attribute prediction, so we assume a strong model would do even better
cfg.MODEL.ARCH_NAMES = ['resnet18']
cfg.MODEL.PRETRAINED = True
cfg.OUTPUT = CN()
cfg.OUTPUT.SAVE_PATH = os.path.join(base_path,'experiments/Experiment_2_No_Bias_Noise/Experiment_2_Results')



# %%
