
HAM_EXTENSION = 'HAM10k'

DEFAULT_HAM_LABELS = {
    0: ('vasc', 'bkl', 'df', 'nv'),
    1: ('mel','bcc','akiec'),
    -1: ('scc', 'ak')
}

HAM_POSSIBLE_VALS_DICT = {
    'age':[ 0.,  5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.,65., 70., 75., 80., 85.],
    'localization':['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
        'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk',
        'unknown', 'upper extremity'],
    'dataset':['rosendahl', 'vidir_modern', 'vidir_molemax', 'vienna_dias'],
}

HAM_NUM_CATEGORIES = {key:len(val) for key,val in HAM_POSSIBLE_VALS_DICT.items()}


def encode_age(age):
    age_categories = HAM_POSSIBLE_VALS_DICT['age']
    assert age in age_categories, "Encoding for age not found"
    return age_categories.index(age)

def encode_localization(localize):
    localization_categories = HAM_POSSIBLE_VALS_DICT['localization']
    assert localize in localization_categories, "Encoding for localization not found"
    return localization_categories.index(localize)

def encode_dataset(dataset):
    dataset_categories = HAM_POSSIBLE_VALS_DICT['dataset']
    assert dataset in dataset_categories, "Encoding for dataset not found"
    return dataset_categories.index(dataset)


HAM_TARGET_TRANSFORMS = {
    'age': encode_age,
    'sex': lambda x: 1 if x == 'male' else 0,
    'localization':encode_localization,
    'dataset':encode_dataset,
    'fitzpatrick':lambda x: x-1
}