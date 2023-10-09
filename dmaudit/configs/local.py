from yacs.config import CfgNode as CN
import os

cfg = CN()
cfg.SYSTEM = CN()
cfg.SYSTEM.BASEPATH = "C:\\Users\\Mitchell\\Desktop\\dmaudit\\dm-audit\\dmaudit"
cfg.SYSTEM.RAW_DATA_PATH = os.path.join(cfg.SYSTEM.BASEPATH,'data\\Raw_Data')
cfg.SYSTEM.PROCESSED_DATA_PATH =  os.path.join(cfg.SYSTEM.BASEPATH,'data\\Processed_Data')

cfg.SYSTEM.HAM_EXTENSION = "HAM10k"
cfg.SYSTEM.HAM_NATURAL_EXTENSION = os.path.join(cfg.SYSTEM.HAM_EXTENSION, "Natural/HAM10k-resized-256x256")
cfg.SYSTEM.HAM_ARTIFACT_EXTENSION = os.path.join(cfg.SYSTEM.HAM_EXTENSION, "Synthetic")


def get_cfg_locals():
    """
    Get a yacs CfgNode object with local parameters
    """
    # Return a clone so that the local values will not be altered.
    return cfg.clone()
