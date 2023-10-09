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
from tigramite.independence_tests import CMIsymb

from yacs.config import CfgNode as CN
from dmaudit.configs.local import get_cfg_locals

# %%

cfg = get_cfg_locals()

experiment = 'Experiment_0_Mutual'
experiment_path = os.path.join(cfg.SYSTEM.BASEPATH,'experiments',experiment)
train_dataframe_folders = []
res_autoaugment = glob.glob(os.path.join(experiment_path,'Experiment_0_AutoAugment_Results/*'))
res_basic_augment =  glob.glob(os.path.join(experiment_path,'Experiment_0_Results/*'))


train_dataframes_with_info = []
for path in res_basic_augment:
    temp = {}
    file_name = os.path.split(path)[-1]
    n = file_name.split('_')[1]
    ratio = file_name.split('_')[-1]
    temp['ratio'] = ratio 
    temp['augment'] = 'basic'
    temp['n'] = n
    temp['dataframe'] = pd.read_csv(os.path.join(path,'export_all.csv'))
    train_dataframes_with_info.append(temp)

for path in res_autoaugment:
    temp = {}
    file_name = os.path.split(path)[-1]
    n = file_name.split('_')[1]
    ratio = file_name.split('_')[-1]
    temp['ratio'] = ratio 
    temp['augment'] = 'autoaugment'
    temp['n'] = n
    temp['dataframe'] = pd.read_csv(os.path.join(path,'export_all.csv'))
    train_dataframes_with_info.append(temp)
# train_dataframe_info = [{'ratio':i.split('_')}]

#%%
from sklearn.metrics import roc_auc_score,precision_score,recall_score
import numpy as np
import scipy.stats as st


def get_mean_and_confidence_interval(data):
    mean = np.mean(data)
    bottom,top = st.t.interval(alpha=0.95, df=len(data)-1, loc=mean, scale=st.sem(data)) 
    plus_minus = (top-bottom)/2
    return mean,plus_minus

def get_pretty_mean_and_confidence_interval(data):
    mean = np.mean(data)
    bottom,top = st.t.interval(alpha=0.95, df=len(data)-1, loc=mean, scale=st.sem(data)) 
    plus_minus = (top-bottom)/2
    return f"{mean:.2g}\u00B1{plus_minus:.2g}"

# all artifacts biased towards mel
# worst case dataset is all mel have no artifact all benign have artifact
#%%
from sklearn.metrics.cluster import adjusted_mutual_info_score
rows_list = []

for index,curr_dict in enumerate(train_dataframes_with_info):
    curr_row = {'ratio':curr_dict['ratio']}
    orig_auc = []
    worst_case_auc = []

    mel_examples = curr_dict['dataframe'].is_malignant == 1.0
    num_art_given_mal = sum(curr_dict['dataframe'][mel_examples].artifact)
    num_art_given_benign = sum(curr_dict['dataframe'][~mel_examples].artifact)
    curr_row['augment'] = train_dataframes_with_info[index]['augment']


    curr_row['p(artifact|malignant)'] = f"{num_art_given_mal/sum(mel_examples):.2g}"
    curr_row['p(artifact|benign)'] = f"{num_art_given_benign/sum(~mel_examples):.2g}"
    
    mi_estimator = CMIsymb()
    artifact_labels = np.array(curr_dict['dataframe'].artifact).reshape((-1,1)).astype(int)
    y_labels = np.array(curr_dict['dataframe'].is_malignant).reshape((-1,1)).astype(int)
    curr_row['mi'],p_value = mi_estimator.run_test_raw(artifact_labels,y_labels)
    curr_row['mi_percentile'] = 1-p_value
    curr_row['ami'] = adjusted_mutual_info_score(artifact_labels.ravel(),y_labels.ravel())

    for k in range(3):
        curr_dataframe = curr_dict['dataframe'][curr_dict['dataframe']['fold'] == k]
        orig_preds = curr_dataframe['pred-is_malignant-swin_t-global_compression_quality_30-orig_dset_ratio']
        nat_preds = curr_dataframe['pred-is_malignant-swin_t-global_compression_quality_30-counterfactual_natural_images']
        art_preds = curr_dataframe['pred-is_malignant-swin_t-global_compression_quality_30-counterfactual_artifact_images']
        worst_case_curr_preds = np.where(curr_dataframe.is_malignant, nat_preds, art_preds)
        orig_auc.append(roc_auc_score(curr_dataframe.is_malignant,orig_preds))
        worst_case_auc.append(roc_auc_score(curr_dataframe.is_malignant,worst_case_curr_preds))
        
    curr_row['raw_AUC'],curr_row['CI_AUC'] = get_mean_and_confidence_interval(orig_auc)
    curr_row['raw_Counterfactual_AUC'],curr_row['CI_Counterfactual_AUC'] = get_mean_and_confidence_interval(worst_case_auc)

    curr_row['AUC'] = get_pretty_mean_and_confidence_interval(orig_auc)
    curr_row['Counterfactual AUC'] = get_pretty_mean_and_confidence_interval(worst_case_auc)

    rows_list.append(curr_row)

results_df = pd.DataFrame(rows_list)

#%%
results_df = results_df.sort_values(by='mi').reset_index()
# results_df[['p(artifact|melanoma)','p(artifact|not melanoma)','mi','mi_percentile','AUC','Counterfactual AUC']].to_latex("./exp_0_bias.tex")
#%%
results_df

# %%
import matplotlib.pyplot as plt
import seaborn as sns
results_df = results_df.sort_values(by='mi').reset_index()
# %%
# ## ------- AUC / Counter AUC vs Mutual Information -------
sns.set_theme(context='paper',font_scale=1.5)
# sns.set_style("ticks")
autoaugment_results_df = results_df[results_df.augment=='autoaugment']
basicaugment_results_df = results_df[results_df.augment=='basic']

plt.errorbar(autoaugment_results_df.mi,
             autoaugment_results_df.raw_AUC, 
             yerr=autoaugment_results_df.CI_AUC,
             marker='s',
             markersize=6,
             label='Train Ratio -- AutoAugment',
             linestyle='--'
             )
plt.errorbar(autoaugment_results_df.mi,
             autoaugment_results_df.raw_Counterfactual_AUC, 
             yerr=autoaugment_results_df.CI_Counterfactual_AUC,
             marker='s',
             markersize=6,
             label='Counterfactual Ratio -- AutoAugment',
             linestyle='--'
             )

plt.errorbar(basicaugment_results_df.mi,
             basicaugment_results_df.raw_AUC, 
             yerr=basicaugment_results_df.CI_AUC,
             marker='s',
             markersize=6,
             label='Train Ratio'
             )
plt.errorbar(basicaugment_results_df.mi,
             basicaugment_results_df.raw_Counterfactual_AUC, 
             yerr=basicaugment_results_df.CI_Counterfactual_AUC,
             marker='s',
             markersize=6,
             label='Counterfactual Ratio'
             )
# plt.fill_between(results_df.mi, 
#                  results_df.AUC-results_df.plus_or_minus,
#                  results_df.AUC+results_df.plus_or_minus,
#                  alpha=.15)
plt.ylabel('AUC',labelpad=15)
plt.xlabel('MI(Artifact, Y)',labelpad=15)
plt.title('Model Bias by Mutual Information',weight='bold')
plt.legend()
sns.despine()
plt.savefig('./AUC_Counterfactual_AUC_vs_MI.eps', format='eps',bbox_inches='tight')






#%%
# def result_to_metric(x_labels,x_name, values,metric_name):
#     return pd.DataFrame({x_name:x_labels,'value':values,'metric':[metric_name]*len(x_labels)})


# metric_name_dict = {
#         'AUC':orig_auc,
#         'Counterfactual Natural AUC':nat_auc,
#         'Counterfactual Artifact AUC':art_auc,
#         'Counterfactual Natural Precision':natural_precision,
#         'Counterfactual Natural Recall':natural_recall,
#         'Counterfactual Artifact Precision':artifact_precision,
#         'Counterfactual Artifact Recall':artifact_recall
#         }
# output_list = [result_to_metric(ratio_labels,'ratio',item,key) for key,item in metric_name_dict.items()]



# output_df = pd.concat(output_list)


# pd.value_counts(train_dataframes_with_info[0]['dataframe'].dx)



# %%
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

def get_mean_and_confidence_interval(data):
    mean = np.mean(data)
    bottom,top = st.t.interval(alpha=0.95, df=len(data)-1, loc=mean, scale=st.sem(data)) 
    plus_minus = (top-bottom)/2
    return mean,plus_minus




rows_list = []
bootstrap_cmi = True
bootstrap_iterations = 1000
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
    
    for k in range(3):
        print('fold',k)
        curr_dataframe = curr_dict['dataframe'][curr_dict['dataframe']['fold'] == k]
        orig_preds = curr_dataframe['pred-artifact-resnet18-HAM10k-resized-256x256-orig_dset_ratio']

        orig_auc.append(roc_auc_score(curr_dataframe.artifact,orig_preds))

    curr_row['AUC'],curr_row['plus_or_minus'] = get_mean_and_confidence_interval(orig_auc)
    
    mi_estimator = CMIsymb()
    artifact_labels = np.array(curr_dict['dataframe'].artifact).reshape((-1,1)).astype(int)
    y_labels = np.array(curr_dict['dataframe'].is_malignant).reshape((-1,1)).astype(int)
    artifact_probs = np.array(curr_dict['dataframe']['pred-artifact-resnet18-HAM10k-resized-256x256-orig_dset_ratio']).reshape((-1,1))
    artifact_predictions = np.round(artifact_probs).astype(int)

    curr_row['mi'],curr_row['mi_p_value'] = mi_estimator.run_test_raw(artifact_labels,y_labels)
    
    cmi_estimator = CMIsymb()


    curr_row['cmi'],curr_row['cmi_p_value'] = mi_estimator.run_test_raw(artifact_labels,artifact_predictions,z=y_labels)
    
    if bootstrap_cmi:
        
        # print(array_for_bootstrap.shape)
        # curr_row['cmi_bootstrap_lower'],curr_row['cmi_bootstrap_upper'] = cmi_estimator.get_bootstrap_confidence(array_for_bootstrap,np.arange(3),conf_samples=1000)
        temp_bootstrap_cmi = []
        for count in range(bootstrap_iterations):
            indices = np.arange(len(artifact_labels))
            indices_selected = gen.choice(indices, size=len(indices),replace=True)
            selected_artifact = artifact_predictions[indices_selected]
            selected_artifact_labels = artifact_labels[indices_selected]
            selected_y_labels = y_labels[indices_selected]
            list_vals = [selected_artifact.reshape((1,-1)), 
                         selected_artifact_labels.reshape((1,-1)),
                         selected_y_labels.reshape((1,-1))]
            
            array_for_bootstrap = np.concatenate(list_vals,axis=0)
            # get cmi without p value. 
            bootstrap_cmi = cmi_estimator.get_dependence_measure(array_for_bootstrap,np.arange(3))
            temp_bootstrap_cmi.append(bootstrap_cmi)
        #
        bootstrap_st_err = np.std(temp_bootstrap_cmi,ddof=1)

        
        
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
# import matplotlib.pyplot as plt
# import seaborn as sns
# results_df.sort_values(by='mi',inplace=True)

# results_df['cmi_percentile'] = 1-results_df.cmi_p_value
# ## ------- AUC vs Mutual Information -------
# sns.set_theme(context='paper',font_scale=1.5)
# # sns.set_style("ticks")
# plt.errorbar(results_df.mi,
#              results_df.AUC, 
#              yerr=results_df.plus_or_minus,
#              marker='s',
#              markersize=6)
# # plt.fill_between(results_df.mi, 
# #                  results_df.AUC-results_df.plus_or_minus,
# #                  results_df.AUC+results_df.plus_or_minus,
# #                  alpha=.15)
# plt.ylabel('Test Set AUC',labelpad=15)
# plt.xlabel('MI(Artifact, Y)',labelpad=15)
# plt.title('Invisible Artifact: AUC vs Mutual Information',weight='bold')

# sns.despine()
# plt.savefig('./Invisible_Artifact_AUC_vs_MI.eps', format='eps',bbox_inches='tight')

#%%------- CMI vs Mutual Information -------
# sns.set_theme(context='paper',font_scale=1.5)
# fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

# bootstrap_means = (results_df.cmi_bootstrap_upper+results_df.cmi_bootstrap_lower)/2
# bootstrap_variance = (results_df.cmi_bootstrap_upper-results_df.cmi_bootstrap_lower)/2

# bias_estimate = bootstrap_means - results_df.cmi
# bias_corrected_cmi_estimate = results_df.cmi - bias_estimate

# ax1.plot(results_df.mi, results_df.cmi, marker='s', markersize=6,label="Original")
# ax1.errorbar(results_df.mi,
#              bias_corrected_cmi_estimate, 
#              yerr=(results_df.cmi_bootstrap_upper-results_df.cmi_bootstrap_lower)/2,
#              marker='s',
#              markersize=6,label="Bootstrap Distribution")
# ax1.axhline(y = 0.0, color = 'b', linestyle = '--', label = "0.0")
# ax1.set_xlabel('MI(Artifact, Y)',labelpad=15)
# ax1.set_ylabel('MI(Artifact, Artifact Prediction | Y)',labelpad=15)
# ax1.title.set_text('CMI Test Statistic vs MI with Bootstrap')
# ax1.legend()


# ax2.scatter(results_df.mi, results_df.cmi_percentile)
# ax2.set_ylim(-.05,1.0)
# # ax2.set_xlim(-.05,max(results_df.mi)+.05)
# ax2.axhline(y = .95, color = 'r', linestyle = '--', label = "95th Percentile")
# # ax2.fill_between([-.05,max(results_df.mi)+.05], .95, 1.0, facecolor='red', alpha=0.5)
# # ax2.legend(loc = 'upper right')
# ax2.text(0.175,.95, "95th Percentile", color="k", horizontalalignment='left',va="center", bbox=dict(boxstyle='round',fc="white",alpha=.8))
# ax2.set_ylabel("Percentile vs Permutted",labelpad=15)
# ax2.set_xlabel("MI(Artifact, Y)",labelpad=15)
# ax2.title.set_text('CMI Test Percentile vs MI')
# plt.suptitle('Invisible Artifact CMI Results',weight='bold')
# %%
