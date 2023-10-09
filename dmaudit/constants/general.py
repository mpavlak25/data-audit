from torchvision import transforms
from timm.data import rand_augment_transform

# move just the paths to a new file in constants called paths
RAW_DATA_PATH = '/Users/mitchell/Desktop/DM_AUDIT/dm-audit/Raw_Data/'

PROCESSED_DATA_PATH = '/Users/mitchell/Desktop/DM_AUDIT/dm-audit/Processed_Data/'

IMNET_STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

IMNET_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(**IMNET_STATS),
])

IMNET_W_RANDAUGMENT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(**IMNET_STATS),
])

IMNET_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(**IMNET_STATS),
])