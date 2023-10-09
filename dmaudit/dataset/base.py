import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
from functools import lru_cache

from dmaudit.constants.general import IMNET_STATS, IMNET_TRAIN_TRANSFORM, IMNET_TEST_TRANSFORM
from dmaudit.constants.ham import HAM_TARGET_TRANSFORMS, DEFAULT_HAM_LABELS


class CsvDataset(Dataset):
    
    def __init__(self,
                 metadata_path,
                 test_folds=None,
                 split='train',
                 image_key='image_id',
                 paths=None,
                 fold_col='fold',
                 select_path_col='artifact',
                 override_select_path_col = None,
                 label_col='label',
                 ext='.jpg',
                 transform=None,
                 return_meta=False):
        """
        metadata_path -- path to csv containing image metadata (experiment dataframe)
        test_folds -- list of folds that should be used for testing
        split -- "train" to use all folds that are not in test_folds, test otherwise 
        image_key -- string name of column in metadata_path with image file names except extension
        paths -- list of paths where each path contains all images in metadata_path. ie [../Natural, ../dark]
        select_path_col -- column in the csv at metadata_path that corresponds to which index in 
        paths the image should be sampled from.
        label_col -- which column to return as y label
        ext -- image extension to be added to all image keys
        transform -- transforms to apply before returning images
        return_meta -- whether or not to return all metadata
        """
        self.paths = list(map(Path, paths))
        self.df = pd.read_csv(metadata_path)
        self.fold_col = fold_col
        self.image_key = image_key
        self.ext = ext
        self.transform = transform
        self.test_folds = test_folds
        self._split = split
        self.return_meta = return_meta
        self.select_path_col = select_path_col
        self.override_select_path_col = override_select_path_col
        self.label_col = label_col
        
        self.update_df_from_split()
    
    def update_df_from_split(self):
        """
        Uses metadata and split information to update the base dataframe.
        """
        if self.test_folds is not None:
            if self._split == 'train':
                valid = ~(self.df[[self.fold_col]].isin(self.test_folds))
            else:
                valid = self.df[[self.fold_col]].isin(self.test_folds)
            self.metadata = self.df[valid[self.fold_col]]
            self.metadata = self.metadata.reset_index()
    
    @property
    def split(self):
        """'train' or 'test'"""
        return self._split
    
    @split.setter
    def split(self, value):
        self._split = value
        self.update_df_from_split()
    
    @split.deleter
    def split(self):
        del self._split
        self.metadata = self.df
    
    @lru_cache(maxsize=20000)
    def load_img(self,path):
        """Image loading with threadsafe cache wrapper."""
        return Image.open(path)
    
    def get_curr_class_counts(self,indices):
        print(pd.value_counts(self.df[self.label_col]))
        return pd.value_counts(self.metadata.iloc[indices][self.label_col])

    def get_targets(self,indices):
        return self.metadata.iloc[indices][self.label_col]

    def __getitem__(self, item):
        row = self.metadata.iloc[item]
        
        if self.override_select_path_col is not None:
            use_path_to_imgs = self.paths[self.override_select_path_col]
        else:
            use_path_to_imgs = self.paths[int(row[self.select_path_col])]
        img_path = os.path.join(use_path_to_imgs, row[self.image_key] + self.ext)
        img = self.load_img(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        target = row[self.label_col]
        if self.return_meta:
            return img, target, row.to_dict()
        
        return img, target
    
    def __len__(self):
        return len(self.metadata)
