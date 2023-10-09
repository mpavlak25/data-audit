
#%%
import pandas as pd
import torch
import os
import numpy as np
from PIL import Image
import sys 
import glob
import tqdm
from dmaudit.configs.local import get_cfg_locals
from dmaudit.constants.ham import DEFAULT_HAM_LABELS

cfg = get_cfg_locals()
RAW_DATA_PATH = cfg.SYSTEM.RAW_DATA_PATH
PROCESSED_DATA_PATH = cfg.SYSTEM.PROCESSED_DATA_PATH
HAM_EXTENSION = cfg.SYSTEM.HAM_EXTENSION




# ensuring all necessary files exist  
root = os.path.join(RAW_DATA_PATH,HAM_EXTENSION)
# print(os.path.join(root,'dataverse_files',"HAM10000_images_part_1/"))
assert os.path.exists(os.path.join(root,'dataverse_files',"HAM10000_images_part_1/")), "Ham10k part one missing"
assert os.path.exists(os.path.join(root,'dataverse_files',"HAM10000_images_part_2/")), "Ham10k part two missing"
assert os.path.exists(os.path.join(root,'dataverse_files',"HAM10000_metadata")), "Ham10k default metadata file missing"
assert os.path.exists(os.path.join(root,'isic_train_20-19-18-17.csv')), "Extra metadata file not present, please download from https://github.com/pbevan1/Skin-Deep-Unlearning/blob/main/data/csv/isic_train_20-19-18-17.csv"

processed_base_path = os.path.join(PROCESSED_DATA_PATH,HAM_EXTENSION)
processed_images_path = os.path.join(processed_base_path,'Natural','HAM10k-resized-256x256')
if not os.path.exists(os.path.join(processed_base_path,'Natural')):
    os.makedirs(os.path.join(processed_base_path,'Natural'))
if not os.path.exists(os.path.join(processed_base_path,'Synthetic')):
    os.makedirs(os.path.join(processed_base_path,'Synthetic'))
if not os.path.exists(processed_images_path):
    os.mkdir(processed_images_path)


default_metadata = pd.read_csv(os.path.join(root,'dataverse_files',"HAM10000_metadata"))
unlearning_metadata = pd.read_csv(os.path.join(root,'isic_train_20-19-18-17.csv'),usecols=['image_name','scale','marked','fitzpatrick'])
unlearning_metadata = unlearning_metadata.rename(columns={'image_name':'image_id'})
full_metadata = default_metadata.join(unlearning_metadata.set_index('image_id'),on='image_id')

#%%
full_metadata['is_malignant'] = (full_metadata.dx.isin(DEFAULT_HAM_LABELS[1.0])).astype(int)

#%%

print("Length before dropping incomplete",len(full_metadata))
full_metadata = full_metadata[~full_metadata.isna().any(axis=1)]
print("Length after dropping incomplete",len(full_metadata))

#%%
# print(full_metadata)
full_metadata = full_metadata.drop_duplicates(subset=['lesion_id'], keep='first')
print(f"Length after dropping duplicates {len(full_metadata)}")
print(f"Malignant after dropping duplicates: {full_metadata.is_malignant.sum()}")
#%%
path_one_image_ids = glob.glob(os.path.join(root,'dataverse_files',"HAM10000_images_part_1/*.jpg"))
path_two_image_ids = glob.glob(os.path.join(root,'dataverse_files',"HAM10000_images_part_2/*.jpg"))

print(f"Length path one ids {len(path_one_image_ids)}, length path two ids {len(path_two_image_ids)}")
#%%
all_img_ids = set(full_metadata.image_id)
path_one_image_ids = [i for i in path_one_image_ids if os.path.splitext(os.path.split(i)[1])[0] in all_img_ids]
path_two_image_ids = [i for i in path_two_image_ids if os.path.splitext(os.path.split(i)[1])[0] in all_img_ids]
print(f"Length after removing duplicates: path one ids {len(path_one_image_ids)}, length path two ids {len(path_two_image_ids)}")
#%%

print('Part 1...')
for image_path in tqdm.tqdm(path_one_image_ids):
    image_name = os.path.split(image_path)[-1]
    resized_image = Image.open(image_path).resize((256,256))
    resized_image.save(os.path.join(processed_images_path,image_name))
print('Part 2...')
for image_path in tqdm.tqdm(path_two_image_ids):
    image_name = os.path.split(image_path)[-1]
    resized_image = Image.open(image_path).resize((256,256))
    resized_image.save(os.path.join(processed_images_path,image_name))  
full_metadata.to_csv(os.path.join(processed_base_path,'HAM10k_Combined_Metadata.csv'))


# %%
