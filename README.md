## Data AUDIT: Identifying Attribute Utility- and Detectability-Induced Bias in Task Models
### MICCAI 2023

We seek to audit at the dataset level to develop targeted hypotheses on the bias downstream models will inherit. We focus on identifying potential shortcuts, and define two metrics we term “utility” and “detectability” respectively. Utility measures how much information knowing an attribute conveys about the task label. For detectability, we seek to measure how well a downstream model could extract the values of the attribute from the image set, excluding task related information. 


#### MICCAI 2023 paper link: https://link.springer.com/chapter/10.1007/978-3-031-43898-1_43

#### Code Information:

Please note, full results require training of several hundred models. For your convenience, the codebase is split into separate config files to recreate each experiment. To run, use for example: `python ./train_test.py --experiment_cfg_path  C:/path/to/repo/dmaudit/experiments/Experiment_0_Mutual/experiment_0_config_autoaugment.py`

More details and further code cleaning coming shortly.  


