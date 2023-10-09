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
experiment = 'Experiment_2_No_Bias'
experiment_path = os.path.join(cfg.SYSTEM.BASEPATH,'experiments',experiment)
default_cfg = get_cfg_defaults()
#%%
train_df_dict['dataframe'][train_df_dict['dataframe'].is_mel==0]
#%%
train_df_dict = {'ratio':'1,00',
    'dataframe':pd.read_csv(os.path.join(experiment_path,'Experiment_2_Results/export_all.csv')),
    'n':1000}
train_df_dict['dataframe'].columns

pred_columns = ['pred-artifact-resnet18-global_compression_quality_20-orig_dset_ratio',
       'pred-artifact-resnet18-global_compression_quality_30-orig_dset_ratio',
       'pred-artifact-resnet18-global_compression_quality_40-orig_dset_ratio',
       'pred-artifact-resnet18-global_compression_quality_50-orig_dset_ratio',
       'pred-artifact-resnet18-global_compression_quality_60-orig_dset_ratio',
       'pred-artifact-resnet18-global_compression_quality_70-orig_dset_ratio',
       'pred-artifact-resnet18-global_compression_quality_80-orig_dset_ratio',
       'pred-artifact-resnet18-global_compression_quality_90-orig_dset_ratio']
qualities = [20,30,40,50,60,70,80,90]


def get_mean_and_confidence_interval(data):
    mean = np.mean(data)
    bottom,top = st.t.interval(alpha=0.95, df=len(data)-1, loc=mean, scale=st.sem(data)) 
    plus_minus = (top-bottom)/2
    return mean,plus_minus

bootstrap_cmi = True
bootstrap_iterations = 1000
gen = np.random.default_rng(seed=default_cfg.SYSTEM.SEED)
rows_list = []
for index,col in enumerate(pred_columns):
    print(index)
    curr_row = {'quality':qualities[index]}
    curr_aucs = []

    for k in range(3):
        print('fold',k)
        curr_dataframe = train_df_dict['dataframe'][train_df_dict['dataframe']['fold'] == k]
        preds = curr_dataframe[col]
        curr_aucs.append(roc_auc_score(curr_dataframe.artifact, preds))
    curr_row['auc'],curr_row['+/- auc'] = get_mean_and_confidence_interval(curr_aucs)


    artifact_labels = np.array(train_df_dict['dataframe'].artifact).reshape((-1,1)).astype(int)
    y_labels = np.array(train_df_dict['dataframe'].is_mel).reshape((-1,1)).astype(int)
    artifact_probs = np.array(train_df_dict['dataframe'][col]).reshape((-1,1))
    artifact_predictions = np.round(artifact_probs).astype(int)

    cmi_estimator = CMIsymb()


    curr_row['cmi'],curr_row['cmi_p_value'] = cmi_estimator.run_test_raw(artifact_labels,artifact_predictions,z=y_labels)
    rows_list.append(curr_row)

results_df = pd.DataFrame(rows_list)

#%%
results_df
results_df.sort_values(by='quality',inplace=True)

results_df['cmi_percentile'] = 1-results_df.cmi_p_value
#%%
# import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import seaborn as sns
# Create 2x2 sub plots

gs = gridspec.GridSpec(2, 2)


plt.figure()
plt.suptitle('CMI Metrics vs AUC Ground Truth',fontweight='bold')
sns.set_theme(context='paper',font_scale=1.25)
ax = plt.subplot(gs[:,0]) # all rows, col 0
ax.set_ylim((0.0,1.05))
plt.plot(results_df.quality,results_df.cmi_percentile,marker='.')
plt.title('CMI Permutation Percentile')
ax.set_xlabel('Compression Quality')
ax.set_ylabel('Percentile')

ax = plt.subplot(gs[0, 1]) # row 0, col 1
# plt.plot([0,1])
ax.set_ylim((0.5,1.05))
ax.set_xticks(qualities)
plt.title('AUC')
ax.set_xlabel('Compression Quality')
ax.set_ylabel('AUC')
ax.set_xticks(qualities)
plt.plot(results_df.quality,results_df.auc,marker='.')
# plt.errorbar(results_df.quality,
#              results_df.auc, 
#              yerr=results_df['+/- auc'],
#              marker='s',
#              markersize=6,label='auc')
plt.fill_between(results_df.quality, 
                 results_df.auc-results_df['+/- auc'],
                 results_df.auc+results_df['+/- auc'],
                 alpha=.25)

ax = plt.subplot(gs[1, 1]) # row 1, span all columns
ax.set_xticks(qualities)
plt.title('CMI Test Statistic')
ax.set_xlabel('Compression Quality')
ax.set_ylabel('CMI')
plt.plot(results_df.quality,results_df.cmi,marker='.',label='CMI')
sns.despine()
plt.tight_layout()
plt.savefig('./CMI_vs_GroundTruth.eps',format='eps',bbox_inches='tight')

