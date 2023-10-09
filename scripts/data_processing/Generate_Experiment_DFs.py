# %%
import glob
import pandas as pd
import numpy as np
import os

from yacs.config import CfgNode as CN
from dmaudit.configs.defaults import combine_cfgs
from dmaudit.configs.local import get_cfg_locals
from dmaudit.constants.ham import HAM_TARGET_TRANSFORMS

# %%
cfg = get_cfg_locals()
BASE_PATH = cfg.SYSTEM.BASEPATH
RAW_DATA_PATH = cfg.SYSTEM.RAW_DATA_PATH
PROCESSED_DATA_PATH = cfg.SYSTEM.PROCESSED_DATA_PATH
HAM_EXTENSION = cfg.SYSTEM.HAM_EXTENSION
# %%


def create_experiment_df(metadata_path,
                         bias_on_column,
                         odds,
                         fixed_n,
                         bias_on_value=1.0,
                         use_p_a_given_y = False,
                         gen=np.random.default_rng(seed=123),
                         k_folds=3):
    """
    metadata_path -- string path to the 
    bias_on_column -- column we should add artifacts in a bias manner based on (ie y)
    ratio -- two element list giving the ratio of percent artifact positive cases given bias_on_column = 1 to percent artifact 
             positive given bias_on_column = 0
    bias_on_value -- value of a positive case
    fixed_n -- if none uses only ratio, otherwise simplifies ratio and samples a total number of fixed_n cases
    """
    
    metadata_df = pd.read_csv(metadata_path)
    
    total_y_true = len(metadata_df[metadata_df[bias_on_column] == bias_on_value])
    total_y_false = len(metadata_df) - total_y_true
    # scale ratio incase it doesn't add to 1
    # print(total_y_true,total_y_false)
    
    if odds == 0:
        temp_a = 0
        temp_b = 1
    
    elif odds == -1:
        temp_a = 1
        temp_b = 0
    else:
        temp_a = odds * 1
        temp_b = 1
    pos_ratio = temp_a / (temp_a + temp_b)
    neg_ratio = temp_b / (temp_a + temp_b)
    
    if use_p_a_given_y:
        scaling = fixed_n / (pos_ratio * total_y_true + neg_ratio * total_y_false)
        pos_factor, neg_factor = pos_ratio * scaling, neg_ratio * scaling
        if pos_factor == 0 or neg_factor == 0:
            sampling_pos = pos_factor
            sampling_negative = neg_factor
        else:
            sampling_pos = (pos_factor * fixed_n) / (neg_factor * total_y_false + pos_factor * total_y_true)
            sampling_negative = (sampling_pos * neg_factor) / pos_factor
        
        assert sampling_pos < 1.0, "Ratio does not work given dataset statistics"
        assert sampling_negative < 1.0, "Ratio does not work given dataset statistics"
        
        total_artifact_y_true = sampling_pos * total_y_true
        total_artifact_y_false = sampling_negative * total_y_false
    else:
        total_artifact_y_true = int(pos_ratio * fixed_n)
        total_artifact_y_false = fixed_n - total_artifact_y_true
        assert total_artifact_y_true >= 0 and total_artifact_y_true <=total_y_true
        assert total_artifact_y_false >= 0 and total_artifact_y_false <= total_artifact_y_false

    
    all_artifacts = np.zeros((len(metadata_df)))
    
    pos_indices = np.arange(len(metadata_df))[metadata_df[bias_on_column] == bias_on_value]
    neg_indices = np.arange(len(metadata_df))[metadata_df[bias_on_column] != bias_on_value]
    
    selected_pos = gen.choice(pos_indices, int(total_artifact_y_true), replace=False)
    selected_neg = gen.choice(neg_indices, int(total_artifact_y_false), replace=False)
    print(len(selected_pos),len(selected_neg))    
    all_artifacts[selected_pos] = 1
    all_artifacts[selected_neg] = 1
    metadata_df['artifact'] = all_artifacts
    if k_folds:
        metadata_df['fold'] = gen.choice(k_folds, len(metadata_df), replace=True)
    return metadata_df
#%%
# df = pd.read_csv(ham_df_path)
# print(sum(df.dx == 'akiec'),sum(df.dx=='mel'),sum(df.dx == 'bcc'))


# %%
ham_processed_base_path = os.path.join(PROCESSED_DATA_PATH, HAM_EXTENSION)
ham_df_path = os.path.join(ham_processed_base_path, 'HAM10k_Combined_Metadata.csv')
# %%
#Experiment 0
odds = [0.18666,.5, 1,3,9,-1]
number_samples = [1000]
out_path = os.path.join(BASE_PATH, 'experiments/Experiment_0_Mutual/Experiment_0_Train_DFs')
# For synthetic scaling experiment
if not os.path.exists(out_path):
    os.mkdir(out_path)

for n in number_samples:
    for curr_odds in odds:
        curr_df = create_experiment_df(ham_df_path,
                                       bias_on_column='is_malignant',
                                       odds=curr_odds,
                                       bias_on_value=1.0,
                                       fixed_n=n)
        # drop unnamed column
        curr_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        output_name = f'fixedn_{n}_ham10k_oddsratio_{curr_odds:.2f}.csv'
        output_path = os.path.join(out_path, output_name)
        curr_df.to_csv(output_path)

# %%
# Experiment 1
odds = [0, 1/9, 0.18666, 1/2, 1, 2, 4, 9, -1]
number_samples = [1000]
out_path = os.path.join(BASE_PATH, 'experiments/Experiment_1_Invisible_Artifact/Experiment_1_Train_DFs')
# For synthetic scaling experiment
if not os.path.exists(out_path):
    os.mkdir(out_path)

for n in number_samples:
    for curr_odds in odds:
        curr_df = create_experiment_df(ham_df_path,
                                       bias_on_column='is_malignant',
                                       odds=curr_odds,
                                       bias_on_value=1.0,
                                       fixed_n=n)
        # drop unnamed column
        curr_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        output_name = f'fixedn_{n}_ham10k_oddsratio_{curr_odds:.2f}.csv'
        output_path = os.path.join(out_path, output_name)
        curr_df.to_csv(output_path)

# %%
# Experiment 2
# ratio gives 157/1162 malignant and 843/6225
odds = [0.18666]
number_samples = [1000]
out_path = os.path.join(BASE_PATH, 'experiments/Experiment_2_No_Bias/Experiment_2_Train_DFs')
# For synthetic scaling experiment
if not os.path.exists(out_path):
    os.mkdir(out_path)

for n in number_samples:
    for curr_odds in odds:
        curr_df = create_experiment_df(ham_df_path,
                                       bias_on_column='is_malignant',
                                       odds=curr_odds,
                                       bias_on_value=1.0,
                                       fixed_n=n)
        # drop unnamed column
        curr_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        output_name = f'fixedn_{n}_ham10k_oddsratio_{curr_odds:.2f}.csv'
        output_path = os.path.join(out_path, output_name)
        curr_df.to_csv(output_path)


# %%
# Experiment 3
odds = [0.18666]
number_samples = [50,250,500,750,1000]
out_path = os.path.join(BASE_PATH, 'experiments/Experiment_3_Scaling/Experiment_3_Train_DFs')
# For synthetic scaling experiment
if not os.path.exists(out_path):
    os.mkdir(out_path)

for n in number_samples:
    for curr_odds in odds:
        curr_df = create_experiment_df(ham_df_path,
                                       bias_on_column='is_malignant',
                                       odds=curr_odds,
                                       bias_on_value=1.0,
                                       fixed_n=n)
        # drop unnamed column
        curr_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        output_name = f'fixedn_{n}_ham10k_oddsratio_{curr_odds:.2f}.csv'
        output_path = os.path.join(out_path, output_name)
        curr_df.to_csv(output_path)

# %%
# Experiment 4
odds = [0.18666,.666666, 1, 2, 4, 9, -1]
number_samples = [1000] 
out_path = os.path.join(BASE_PATH, 'experiments/Experiment_4_MI_vs_CMI/Experiment_4_Train_DFs')
# For synthetic scaling experiment
if not os.path.exists(out_path):
    os.mkdir(out_path)

for n in number_samples:
    for curr_odds in odds:
        curr_df = create_experiment_df(ham_df_path,
                                       bias_on_column='is_malignant',
                                       odds=curr_odds,
                                       bias_on_value=1.0,
                                       fixed_n=n)
        # drop unnamed column
        curr_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        output_name = f'fixedn_{n}_ham10k_oddsratio_{curr_odds:.2f}.csv'
        output_path = os.path.join(out_path, output_name)
        curr_df.to_csv(output_path)

#%%
# Experiment 5 -- Natural Data
# 

def create_natural_df(metadata_path,
                         attribute,
                         gen=np.random.default_rng(seed=123),
                         k_folds=3):
    """
    metadata_path -- string path to the met
    """
    
    metadata_df = pd.read_csv(metadata_path)
    if attribute in HAM_TARGET_TRANSFORMS:
        to_predict_col = metadata_df[attribute].apply(HAM_TARGET_TRANSFORMS[attribute])
    else:
        to_predict_col = metadata_df[attribute]
    metadata_df['attribute'] = to_predict_col
    if k_folds:
        metadata_df['fold'] = gen.choice(k_folds, len(metadata_df), replace=True)
    return metadata_df


#%%
out_path = os.path.join(BASE_PATH, 'experiments/Experiment_5_Natural/Experiment_5_Train_DFs')
# For synthetic scaling experiment
if not os.path.exists(out_path):
    os.mkdir(out_path)

natural_attributes =  ['sex','scale','marked','is_malignant'] + ['age','localization','dataset','fitzpatrick']

for attr in natural_attributes:
        curr_df = create_natural_df(ham_df_path, attr)
        # drop unnamed column
        curr_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        output_name = f'{attr}_ham10k.csv'
        output_path = os.path.join(out_path, output_name)
        curr_df.to_csv(output_path)


# %%


def create_experiment_6_df(metadata_path,
                         bias_on_column,
                         odds,
                         pos_classes,
                         neg_classes,
                         fixed_n,
                         bias_on_value=1.0,
                         use_p_a_given_y = False,
                         gen=np.random.default_rng(seed=123),
                         k_folds=3):
    """
    metadata_path -- string path to the 
    bias_on_column -- column we should add artifacts in a bias manner based on (ie y)
    ratio -- two element list giving the ratio of percent artifact positive cases given bias_on_column = 1 to percent artifact 
             positive given bias_on_column = 0
    bias_on_value -- value of a positive case
    fixed_n -- if none uses only ratio, otherwise simplifies ratio and samples a total number of fixed_n cases
    """
    
    metadata_df = pd.read_csv(metadata_path)
    
    total_y_true = len(metadata_df[metadata_df[bias_on_column] == bias_on_value])
    total_y_false = len(metadata_df) - total_y_true
    # scale ratio incase it doesn't add to 1
    # print(total_y_true,total_y_false)
    
    if odds == 0:
        temp_a = 0
        temp_b = 1
    
    elif odds == -1:
        temp_a = 1
        temp_b = 0
    else:
        temp_a = odds * 1
        temp_b = 1
    pos_ratio = temp_a / (temp_a + temp_b)
    neg_ratio = temp_b / (temp_a + temp_b)

    total_artifact_y_true = int(pos_ratio * fixed_n)
    total_artifact_y_false = fixed_n - total_artifact_y_true
    assert total_artifact_y_true >= 0 and total_artifact_y_true <=total_y_true
    assert total_artifact_y_false >= 0 and total_artifact_y_false <= total_artifact_y_false

    
    all_artifacts = np.zeros((len(metadata_df)))
    
    pos_indices = np.arange(len(metadata_df))[metadata_df[bias_on_column] == bias_on_value]
    neg_indices = np.arange(len(metadata_df))[metadata_df[bias_on_column] != bias_on_value]
    
    selected_pos = gen.choice(pos_indices, int(total_artifact_y_true), replace=False)
    selected_neg = gen.choice(neg_indices, int(total_artifact_y_false), replace=False)
    print(len(selected_pos),len(selected_neg))    
    #mel positive and artifact
    pos_sections = np.array_split(selected_pos,len(pos_classes))
    #mel negative and artifact
    neg_sections = np.array_split(selected_neg,len(pos_classes))
    for index,pos_index in enumerate(pos_classes):
        # print(pos_index,'pos')
        all_artifacts[pos_sections[index]] = pos_index
    for index,pos_index in enumerate(pos_classes):
        # print(neg_index,'neg')
        all_artifacts[neg_sections[index]] = pos_index 

    #benign associated artifact selection
    benign_artifacts_indices = np.nonzero(all_artifacts == 0.0)
    benign_art_sections = np.array_split(benign_artifacts_indices,len(neg_classes))
    for index, benign_art_index in enumerate(benign_art_sections):
        all_artifacts[benign_art_sections[index]] = benign_art_index

    print(pd.value_counts(all_artifacts))
    metadata_df['artifact'] = all_artifacts
    if k_folds:
        metadata_df['fold'] = gen.choice(k_folds, len(metadata_df), replace=True)
    return metadata_df

# %%
# Experiment 6
odds = [0.18666,.666666, 1, 2, 4, 9, -1]
number_samples = [1000] 
out_path = os.path.join(BASE_PATH, 'experiments/Experiment_6_Multiple/Experiment_6_Train_DFs')
# For synthetic scaling experiment
if not os.path.exists(out_path):
    os.mkdir(out_path)

for n in number_samples:
    for curr_odds in odds:
        for class_info in [([1],[0]),([2,3],[0,1]),([4,5,6,7],[0,1,2,3])]:
            pos_classes,neg_classes = class_info[0],class_info[1]
            curr_df = create_experiment_6_df(ham_df_path,
                                        bias_on_column='is_malignant',
                                        odds=curr_odds,
                                        pos_classes=pos_classes,
                                        neg_classes=neg_classes,
                                        bias_on_value=1.0,
                                        fixed_n=n)
            # drop unnamed column
            curr_df.drop(columns=['Unnamed: 0'], inplace=True)
            output_name = f'fixedn_{n}_ham10k_oddsratio_{curr_odds:.2f}_nclasses_{len(pos_classes)+len(neg_classes)}.csv'
            output_path = os.path.join(out_path, output_name)
            curr_df.to_csv(output_path)
# %%
