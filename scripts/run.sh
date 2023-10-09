#!/bin/bash
echo "Start Runs"
python ./train_test.py --experiment_cfg_path  C:/Users/Mitchell/Desktop/dmaudit/dm-audit/dmaudit/experiments/Experiment_0_Mutual/experiment_0_config_autoaugment.py
python ./train_test.py --experiment_cfg_path  C:/Users/Mitchell/Desktop/dmaudit/dm-audit/dmaudit/experiments/Experiment_4_MI_vs_CMI/experiment_4_config_artifact.py
python ./train_test.py --experiment_cfg_path  C:/Users/Mitchell/Desktop/dmaudit/dm-audit/dmaudit/experiments/Experiment_4_MI_vs_CMI/experiment_4_config_dx_autoaugment.py
python ./train_test.py --experiment_cfg_path  C:/Users/Mitchell/Desktop/dmaudit/dm-audit/dmaudit/experiments/Experiment_4_MI_vs_CMI/experiment_4_config_dx.py
# 
# python ./train_test.py --experiment_cfg_path  C:/Users/Mitchell/Desktop/dmaudit/dm-audit/dmaudit/experiments/Experiment_1_Invisible_Artifact/experiment_1_config.py
# python ./train_test.py --experiment_cfg_path  C:/Users/Mitchell/Desktop/dmaudit/dm-audit/dmaudit/experiments/Experiment_2_No_Bias/experiment_2_config.py
wait 5