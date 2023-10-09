# %%
import os
import warnings
from pathlib import Path
from yacs.config import CfgNode as CN

# %%
## This file represents the default configuration for all experiments in this 
## repo. All experiments overwrite this file.

# main config node
cfg = CN()

# System Parameters config node
cfg.SYSTEM = CN()
#   not currently implemented for paralelized execution, but we plan to.
cfg.SYSTEM.NUM_GPUS = 1
#   Number workers -- debug with windows
cfg.SYSTEM.NUM_WORKERS = 4

cfg.SYSTEM.SEED = 123

# must override in local
cfg.SYSTEM.BASEPATH = None
cfg.SYSTEM.RAW_DATA_PATH = None
cfg.SYSTEM.PROCESSED_DATA_PATH = None
cfg.SYSTEM.HAM_EXTENSION = None

cfg.SYSTEM.HAM_NATURAL_EXTENSION = None
cfg.SYSTEM.HAM_ARTIFACT_EXTENSION = None

#

# Training parameters config node
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 128
cfg.TRAIN.NUM_EPOCHS = 10
cfg.TRAIN.OPTIMIZER = 'AdamW'
cfg.TRAIN.LR = 5e-5
cfg.TRAIN.WEIGHT_DECAY = .01
cfg.TRAIN.MOMENTUM = (.9,.999)
cfg.TRAIN.CRITERION = 'ce'
cfg.TRAIN.BINARY_TARGET = True
cfg.TRAIN.RUN_ARTIFACT = True
cfg.TRAIN.RUN_Y_PREDICTION = True

cfg.LOG_PER_N_BATCHES = 10

cfg.TRAIN.SCHEDULER = 'LINEAR_LR_DECAY'
cfg.TRAIN.SCHEDULER_GAMMA = .7
#   Break after a single pass
cfg.TRAIN.DO_DRY_RUN = False
#   Skips training and only evaluates
cfg.TRAIN.EVAL_ONLY = False
cfg.TRAIN.VALIDATION_INTERVAL_N_STEPS = 100
cfg.TRAIN.PROFILE = False
cfg.TRAIN.BALANCE_CLASSES = True

cfg.EXPERIMENT = CN()
# override
cfg.EXPERIMENT.NAME = None

# Data Parameters
cfg.DATASET = CN()
cfg.DATASET.NAME = 'HAM10k'
cfg.DATASET.RESIZE_SHAPE = (224, 224)
cfg.DATASET.K_FOLDS = 3
cfg.DATASET.AUGMENTATION = 'basic'

# Dataframe for experiment runs. Contains pregenerated biased datasets for easy analysis
cfg.DATASET.RUN_DFS = CN()
cfg.DATASET.RUN_DFS.FOLDER_PATH = None
cfg.DATASET.RUN_DFS.FILENAMES = None
#   must override in experiment, list of tuple of paths to datasets that will be used for 
#   each experiment respectively

#   we use this column to determine which path in paths list an image 
#   should be drawn from, allowing us to keep and access pre-processed
#   artifact or natural images on the fly (values correspond to index in data sources)
cfg.DATASET.RUN_DFS.SELECT_IMAGES_PATH_COLUMN = 'artifact'
#   experiment prediction target
cfg.DATASET.RUN_DFS.ARTIFACT_COLUMN = None
cfg.DATASET.RUN_DFS.Y_LABEL_COLUMN = None

cfg.DATASET.RUN_DFS.IMAGE_NAME_COLUMN = 'image_id'
cfg.DATASET.RUN_DFS.FOLD_COLUMN = 'fold'
cfg.DATASET.ARTIFICIAL_SETS_TO_RUN = None
# of the data in train based on our cross validation approach, what percent to train on and 
# what percent to use for validation / tracking metrics
cfg.DATASET.TRAIN_VAL_PERCENTAGES = [.9, .1]
cfg.MODEL = CN()
cfg.MODEL.ARCH_NAMES = []
cfg.MODEL.WEIGHTS_PATH = None
cfg.MODEL.PRETRAINED = True
cfg.MODEL.NUM_LAST_LAYER_OUTPUTS = 2

cfg.TEST = CN()
cfg.TEST.BATCH_SIZE = 128
cfg.TEST.BINARY_TARGET = True
cfg.TEST.DO_DRY_RUN = False
cfg.TEST.COUNTERFACTUAL_INDEX = None 

cfg.OUTPUT = CN()
cfg.OUTPUT.SAVE_PATH = None
cfg.OUTPUT.COMBINE_BY_DF = True
cfg.OUTPUT.SAVE_BEST_MODEL = True


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return cfg.clone()


def combine_cfgs(path_cfg_experiment: Path = None, path_cfg_local: Path = None):
    """
    An internal facing routine that combines CFGs in the order provided.
    :param path_cfg_experiment: path to experiment files that control details for a given experiment
    :param path_cfg_local: local file that contains paths of importance
    :return: cfg_base incorporating the overwrite.
    """
    print(path_cfg_experiment, 'exp')
    print(path_cfg_local, 'local')
    if path_cfg_experiment is not None:
        path_cfg_experiment = Path(path_cfg_experiment)
    if path_cfg_local is not None:
        path_cfg_local = Path(path_cfg_local)
    
    # Path order of precedence is:
    # Priority 1, 2, 3 respectively
    # experiment cfg node > local python > default values
    
    # Load default lowest tier one:
    # Priority 4:
    cfg_base = get_cfg_defaults()
    
    # Merge local cfg_path file to allow machine specific values
    # Priority 2:
    if path_cfg_local is not None:
        assert path_cfg_local.exists(), "Local python config missing"
        cfg_base.merge_from_file(path_cfg_local.absolute())
    
    # Merge experiment specific config
    # Priority 3:
    if path_cfg_experiment is not None:
        assert path_cfg_experiment.exists(), "Make sure specified experiment config python file exists."
        cfg_base.merge_from_file(path_cfg_experiment.absolute())
    
    return cfg_base

# %%
