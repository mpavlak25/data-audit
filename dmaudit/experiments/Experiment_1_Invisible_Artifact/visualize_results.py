# %%
import pandas as pd
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as st
import glob
import os
# from tigramite.independence_tests import CMIsymb

from yacs.config import CfgNode as CN
from dmaudit.configs.local import get_cfg_locals
from dmaudit.configs.defaults import get_cfg_defaults

from sklearn.metrics import roc_auc_score,precision_score,recall_score
import numpy as np
import scipy.stats as st
import knncmi
from tigramite.independence_tests import CMIsymb

# %%
cfg = get_cfg_locals()
experiment = 'Experiment_1_Invisible_Artifact'
experiment_path = os.path.join(cfg.SYSTEM.BASEPATH,'experiments',experiment)
default_cfg = get_cfg_defaults()
#%%

train_dataframe_folders = glob.glob(os.path.join(experiment_path,'Experiment_1_Results/*'))

train_dataframes_with_info = []
for path in train_dataframe_folders:
    temp = {}
    file_name = os.path.split(path)[-1]
    n = file_name.split('_')[1]
    ratio = file_name.split('_')[-1]
    temp['ratio'] = ratio 
    temp['n'] = n
    temp['dataframe'] = pd.read_csv(os.path.join(path,'export_all.csv'))
    train_dataframes_with_info.append(temp)



#%%
temp_df = train_dataframes_with_info[4]['dataframe']
sum(temp_df[temp_df.is_malignant==0].artifact)
#%%
from sklearn.metrics import adjusted_mutual_info_score
def conditional_adjusted_mi(x,y,z,n_z_symbols = None):
    n_z_symbols = z.max() 
    if 'int' not in str(z.dtype):
        warnings.warn("Function requires symbolic conditioning variable. Will be converted to ints")
        z = z.astype(int) 
    adjusted_cami = 0
    values, counts = np.unique(z,return_counts=True)
    counts_dict = {value:counts[index] for index,value in enumerate(values)}
    for z_class in range(n_z_symbols):
        if z_class not in counts_dict:
            continue 
        else:
            p_z = counts_dict[z_class] / sum(counts)
            z_equals_class_indices = z == z_class
            ami = adjusted_mutual_info_score(x[ z_equals_class_indices],y[ z_equals_class_indices])
            adjusted_cami += p_z * ami 

    return adjusted_cami
#%%

def get_mean_and_confidence_interval(data):
    mean = np.mean(data)
    bottom,top = st.t.interval(alpha=0.95, df=len(data)-1, loc=mean, scale=st.sem(data)) 
    plus_minus = (top-bottom)/2
    return mean,plus_minus



rows_list = []
bootstrap_cmi = True
bootstrap_iterations = 10000
# sample_percent = .9
gen = np.random.default_rng(seed=default_cfg.SYSTEM.SEED)
     

for index,curr_dict in enumerate(train_dataframes_with_info):
    print(f"Starting set {index}")
    curr_row = {'ratio':curr_dict['ratio']}
    orig_auc = []
    nat_auc = []
    art_auc = []
    natural_precision = []
    natural_recall = []
    artifact_precision = []
    artifact_recall = []
    rounded_auc = []
    
    for k in range(3):
        print('fold',k)
        curr_dataframe = curr_dict['dataframe'][curr_dict['dataframe']['fold'] == k]
        orig_preds = curr_dataframe['pred-artifact-resnet18-HAM10k-resized-256x256-orig_dset_ratio']

        orig_auc.append(roc_auc_score(curr_dataframe.artifact,orig_preds))
        rounded_auc.append(roc_auc_score(curr_dataframe.artifact,np.round(orig_preds)))

    curr_row['AUC'],curr_row['plus_or_minus'] = get_mean_and_confidence_interval(orig_auc)
    

    mi_estimator = CMIsymb()
    artifact_labels = np.array(curr_dict['dataframe'].artifact).reshape((-1,1)).astype(int)
    y_labels = np.array(curr_dict['dataframe'].is_malignant).reshape((-1,1)).astype(int)
    artifact_probs = np.array(curr_dict['dataframe']['pred-artifact-resnet18-HAM10k-resized-256x256-orig_dset_ratio']).reshape((-1,1))
    artifact_predictions = np.round(artifact_probs).astype(int)

    curr_row['mi'],curr_row['mi_p_value'] = mi_estimator.run_test_raw(artifact_labels,y_labels)
    
    cmi_estimator = CMIsymb()


    curr_row['cmi'],curr_row['cmi_p_value'] = mi_estimator.run_test_raw(artifact_labels,artifact_predictions,z=y_labels)
    curr_row['cami'] = conditional_adjusted_mi(artifact_labels,artifact_predictions,y_labels,n_z_symbols=2)
   

    if bootstrap_cmi:
        bootstrapped_cmi = []
        for iteration in range(bootstrap_iterations):
            if iteration %1000 == 0:
                print(f"curr iteration {iteration}")
            indices = np.arange(len(artifact_labels))
            indices_selected = gen.choice(indices, size=len(indices),replace=True)
            selected_artifact = artifact_predictions[indices_selected]
            selected_artifact_labels = artifact_labels[indices_selected]
            selected_y_labels = y_labels[indices_selected]
            list_vals = [selected_artifact.reshape((1,-1)), 
                        selected_artifact_labels.reshape((1,-1)),
                        selected_y_labels.reshape((1,-1))]
            
            array_for_bootstrap = np.concatenate(list_vals,axis=0)
            # print(array_for_bootstrap.shape)
            bootstrapped_cmi.append(cmi_estimator.get_dependence_measure(array_for_bootstrap,np.arange(3)))
        curr_row['cmi_pm'] = 1.96 * np.std(bootstrapped_cmi,ddof=1)
        

        
        
    # alternate very slow mixed estimator we can use to validate rounding does not remove too much info,
    # too slow to use for getting percentiles

    # curr_row['raw_preds_cmi'] = knncmi.cmi(x=['artifact'],
    #                                        y=['pred-artifact-resnet18-HAM10k-resized-256x256-orig_dset_ratio'],
    #                                        z=['is_malignant'],
    #                                        k=3,
    #                                        data=curr_dict['dataframe'])
    rows_list.append(curr_row)

results_df = pd.DataFrame(rows_list)

#%%

#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

results_df.sort_values(by='mi',inplace=True)

results_df['cmi_percentile'] = 1-results_df.cmi_p_value



## ------- AUC vs Mutual Information -------
gs = gridspec.GridSpec(2, 3)


plt.figure(figsize=(12,4))
plt.suptitle('Invisible Artifact: AUC vs CMI',fontweight='bold')

sns.set_theme(context='paper',font_scale=1.25)
ax = plt.subplot(gs[:,0]) # all rows, col 0,1


# sns.set_style("ticks")
plt.errorbar(results_df.mi,
             results_df.AUC, 
             yerr=results_df.plus_or_minus,
             marker='s',
             markersize=6)
# plt.fill_between(results_df.mi, 
#                  results_df.AUC-results_df.plus_or_minus,
#                  results_df.AUC+results_df.plus_or_minus,
#                  alpha=.15)
plt.ylabel('Test Fold AUC',labelpad=15)
plt.xlabel('MI(Artifact, Y)',labelpad=15)
plt.title('AUC vs Relationship Strength')

# sns.despine()
# plt.savefig('./Invisible_Artifact_AUC_vs_MI.eps', format='eps',bbox_inches='tight')

# import matplotlib.ticker as mtick
# sns.set_theme(context='paper',font_scale=1.5)
# fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,6))
ax1 = plt.subplot(gs[:,1])
ax2 = plt.subplot(gs[:,2])
# bootstrap_means = (results_df.cmi_bootstrap_upper+results_df.cmi_bootstrap_lower)/2
# bootstrap_variance = (results_df.cmi_bootstrap_upper-results_df.cmi_bootstrap_lower)/2

# bias_estimate = bootstrap_means - results_df.cmi
# bias_corrected_cmi_estimate = results_df.cmi - bias_estimate

ax1.errorbar(results_df.mi,
             results_df.cami, 
             yerr=results_df.cmi_pm,
             marker='s',
             markersize=6)

ax1.set_xlabel('MI(Artifact, Y)',labelpad=15)
ax1.set_ylabel(r'CMI',labelpad=15)
ax1.title.set_text('CMI vs MI with Bootstrap')

ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
ax1.axhline(y = 0.0, color = 'k', linestyle = '-', label = "0.0")

ax2.scatter(results_df.mi, results_df.cmi_percentile)
ax2.set_ylim(-.05,1.0)
# ax2.set_xlim(-.05,max(results_df.mi)+.05)
ax2.axhline(y = .95, color = 'r', linestyle = '--', label = "95th Percentile")
# ax2.fill_between([-.05,max(results_df.mi)+.05], .95, 1.0, facecolor='red', alpha=0.5)
# ax2.legend(loc = 'upper right')
ax2.text(0.13,.95, "95th Percentile", color="k", horizontalalignment='left',va="center", bbox=dict(boxstyle='round',fc="white",alpha=.8))
ax2.set_ylabel("CMI Percentile",labelpad=15)
ax2.set_xlabel("MI(Artifact, Y)",labelpad=15)
ax2.title.set_text('Permutation Percentile vs MI')
# plt.suptitle('B) Invisible Artifact CMI Results',weight='bold')
plt.tight_layout()
plt.savefig('./CMI_vs_MI.eps',format='eps',bbox_inches='tight')

# %%


