# %%
import numpy as np
from PIL import Image
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

import cv2
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image

from kornia.filters import gaussian_blur2d, get_gaussian_kernel2d
# from albumentations.augmentations.transforms import ImageCompression
from albumentations.augmentations.functional import iso_noise
from dmaudit.configs.local import get_cfg_locals
from functools import partial

local_cfg = get_cfg_locals()
RAW_DATA_PATH = local_cfg.SYSTEM.RAW_DATA_PATH
PROCESSED_DATA_PATH = local_cfg.SYSTEM.PROCESSED_DATA_PATH
HAM_EXTENSION = local_cfg.SYSTEM.HAM_EXTENSION

PROCESSED_HAM_PATH = os.path.join(PROCESSED_DATA_PATH, HAM_EXTENSION, 'Natural')
EXPECTED_NATURAL_CLASS = 'HAM10k-resized-256x256'

# %%


to_tensor = transforms.ToTensor()
images_list = glob.glob(os.path.join(PROCESSED_HAM_PATH, EXPECTED_NATURAL_CLASS, '*.jpg'))

seed = 1234
torch.random.manual_seed(seed)
np.random.seed(seed)


# def global_jpeg_compression(img, quality):
#     if img.shape[0] == 3:
#         temp_img = torch.tensor(img).permute(1, 2, 0).numpy()
#     else:
#         temp_img = torch.tensor(img).numpy()
#     if temp_img.max() <= 1:
#         temp_img = (temp_img * 255).astype(np.uint8)
#     compressor = ImageCompression(quality_lower=quality, quality_upper=quality, always_apply=True, p=1.0)
#     modded = compressor(image=temp_img.astype(np.uint8))
#     return torch.tensor(modded['image']).permute(2, 0, 1)


def fisheye_distort(img, strength, k_1=.2, k_2=.05):
    img = np.transpose(img.numpy(), (1, 2, 0))
    xs, ys = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    x_center, y_center = img.shape[1] / 2, img.shape[0] / 2
    xs = (xs - x_center) / x_center
    ys = (ys - y_center) / y_center
    radii = np.sqrt(xs ** 2 + ys ** 2)
    
    x_map = (xs / (1 + (k_1 * strength * radii) + (k_2 * strength * (radii ** 2)))).astype("float32")
    y_map = (ys / (1 + (k_1 * strength * radii) + (k_2 * strength * (radii ** 2)))).astype("float32")
    x_map = x_map * x_center + x_center
    y_map = y_map * y_center + y_center
    distorted_im = cv2.remap(img, x_map, y_map, interpolation=cv2.INTER_LINEAR)
    return np.transpose(distorted_im, (2, 0, 1))


def global_hue_distort(img, strength):
    # for strength in self.strengths:
    if img.shape[-1] == 3:
        temp_img = torch.tensor(img).permute(0, 1, 2)
    else:
        temp_img = torch.tensor(img)
    return (TF.adjust_hue(temp_img, strength))


def get_box_start_stop(center, dim=16):
    x_start = center[0] - dim // 2
    x_stop = x_start + dim
    y_start = center[1] - dim // 2
    y_stop = y_start + dim
    return x_start, x_stop, y_start, y_stop


def noise(x_in, center, scale=0.05):
    x_start, x_stop, y_start, y_stop = get_box_start_stop(center)
    x_noise = x_in + scale * torch.randn_like(x_in)
    x_out = x_in + 0.
    x_out[:, y_start:y_stop, x_start:x_stop] = x_noise[:, y_start:y_stop, x_start:x_stop]
    return x_out


def zero(x_in, center):
    x_start, x_stop, y_start, y_stop = get_box_start_stop(center)
    
    x_out = x_in + 0.
    x_out[:, y_start:y_stop, x_start:x_stop] = 0.
    return x_out


def bright(x_in, center, sigma=7, scale=0.5):
    xc, yc = center
    x2 = torch.ones_like(x_in).float()
    k = get_gaussian_kernel2d(kernel_size=(2 * sigma + 1, 2 * sigma + 1),
                              sigma=(sigma, sigma))
    k = 1 + (scale * (k - k.min()) / (k.max() - k.min()))
    start = (xc - k.shape[1] // 2, yc - k.shape[0] // 2)
    stop = (start[0] + k.shape[1], start[1] + k.shape[0])
    x2[:, start[1]:stop[1], start[0]:stop[0]] = k
    mask = x2 > 0
    x_blur = x_in + 0.
    x_blur[mask] = x2[mask] * x_blur[mask]
    return x_blur


def blur(x_in, center, sigma=7, ):
    x_blur = gaussian_blur2d(x_in.unsqueeze(0),
                             kernel_size=(2 * sigma + 1, 2 * sigma + 1),
                             sigma=(sigma, sigma))
    x_blur.squeeze_()
    x_start, x_stop, y_start, y_stop = get_box_start_stop(center)
    x_out = x_in + 0.
    x_out[:, y_start:y_stop, x_start:x_stop] = x_blur[:, y_start:y_stop, x_start:x_stop]
    return x_out


def dark(x_in, center, sigma=7, scale=0.5):
    xc, yc = center
    x2 = torch.ones_like(x_in).float()
    k = get_gaussian_kernel2d(kernel_size=(2 * sigma + 1, 2 * sigma + 1),
                              sigma=(sigma, sigma))
    k = 1 - (scale * (k - k.min()) / (k.max() - k.min()))
    start = (xc - k.shape[1] // 2, yc - k.shape[0] // 2)
    stop = (start[0] + k.shape[1], start[1] + k.shape[0])
    x2[:, start[1]:stop[1], start[0]:stop[0]] = k
    mask = x2 > 0
    x_blur = x_in + 0.
    x_blur[mask] = x2[mask] * x_blur[mask]
    return x_blur

def color_patch(x_in, center, sigma=7, scale=0.5,channels=[0,1]):
    inverse_channels = [i for i in range(3) if i not in channels]
    xc, yc = center
    x2 = torch.ones_like(x_in).float()
    k = get_gaussian_kernel2d(kernel_size=(2 * sigma + 1, 2 * sigma + 1),
                              sigma=(sigma, sigma))
    k_boost = 1 + (scale * (k - k.min()) / (k.max() - k.min()))
    k_reduce =  1 - (scale * (k - k.min()) / (k.max() - k.min()))
    start = (xc - k.shape[1] // 2, yc - k.shape[0] // 2)
    stop = (start[0] + k.shape[1], start[1] + k.shape[0])
    x2[channels, start[1]:stop[1], start[0]:stop[0]] = k_boost
    if inverse_channels:
        x2[inverse_channels, start[1]:stop[1], start[0]:stop[0]] = k_reduce
    mask = x2 > 0
    x_blur = x_in + 0.
    x_blur[mask] = x2[mask] * x_blur[mask]
    return x_blur


def get_box_coords(norm_coord, dims, dim=16):
    c, h, w = dims
    h2, w2 = h // 2, w // 2
    
    norm_coord = min(h2, w2) * norm_coord + min(h2, w2) - dim // 2
    
    x = int(min(max(norm_coord[0], 0), w - dim))
    y = int(min(max(norm_coord[1], 0), h - dim))
    
    xstop = x + dim
    ystop = y + dim
    return x, xstop, y, ystop


def call_box_artifact(x, function, keep_out_pct=.7, dim=16):
    c, h, w = x.shape
    t = 2 * np.pi * np.random.rand()
    u = np.random.rand() + np.random.rand()
    r_out = min(h // 2, w // 2)
    r_in = keep_out_pct * min(h // 2, w // 2)
    r = 2 - u if u > 1 else u
    r = (r_in / r_out) + r * ((r_out - r_in) / r_in) if r < (r_in / r_out) else r
    norm_coord = np.array((r * np.cos(t), r * np.sin(t)))
    x_start, x_stop, y_start, y_stop = get_box_coords(norm_coord, x.shape, dim)
    center = (x_start + dim // 2, y_start + dim // 2)
    return function(x, center)


base_synthetic = os.path.join(PROCESSED_DATA_PATH, HAM_EXTENSION, 'Synthetic')

unmodified_path = os.path.join(base_synthetic, 'No_Artifact')
if not os.path.exists(unmodified_path):
    os.makedirs(unmodified_path)
    print("Saving unmodified images...")
    for path in tqdm(images_list):
        img_id = os.path.split(path)[-1]
        x = Image.fromarray(np.array(Image.open(path)))

        x.save(os.path.join(base_synthetic, f'No_Artifact', f'{img_id}'))

# for compression_quality in [100,99,95, 90, 80,70, 60,50, 40,30, 20]:
#     if not os.path.exists(os.path.join(base_synthetic, f'global_compression_quality_{compression_quality}')):
#         os.makedirs(os.path.join(base_synthetic, f'global_compression_quality_{compression_quality}'))
#     print("Processing compression, quality: ", compression_quality)
#     for index, image_path in tqdm(enumerate(images_list)):
#         img_id = os.path.split(image_path)[-1]
#         x = to_tensor(Image.open(image_path))
#         compressed_img = global_jpeg_compression(x, compression_quality)
#         compressed_img = compressed_img / 255
#         if index == 0:
#             print(compressed_img.mean(), compressed_img.min(), compressed_img.max())
#         img_out = TF.to_pil_image(torch.clip(compressed_img, 0, 1))
#         img_out.save(os.path.join(base_synthetic, f'global_compression_quality_{compression_quality}', f'{img_id}'))

# for fisheye_strength in [.1, .5, 1, 2, 3]:
#     if not os.path.exists(os.path.join(base_synthetic, f'global_fisheye_{fisheye_strength}')):
#         os.makedirs(os.path.join(base_synthetic, f'global_fisheye_{fisheye_strength}'))
#     print("Processing camera distortion, strength: ", fisheye_strength)
#     for index, image_path in tqdm(enumerate(images_list)):
#         img_id = os.path.split(image_path)[-1]
#         x = to_tensor(Image.open(image_path))
#         distort_img = fisheye_distort(x, fisheye_strength)
#         distort_img = torch.tensor(distort_img)
#         # already in 0 to one
#         if index == 0:
#             print(distort_img.mean(), distort_img.min(), distort_img.max())
#         img_out = TF.to_pil_image(torch.clip(distort_img, 0, 1))
#         img_out.save(os.path.join(base_synthetic, f'global_fisheye_{fisheye_strength}', f'{img_id}'))

# for hue_strength in [-.0001, -.0005, -.001, -.05, -.1]:
#     print("Processing hue shift, strength: ", hue_strength)
#     if not os.path.exists(os.path.join(base_synthetic, f'global_hue_{hue_strength}')):
#         os.makedirs(os.path.join(base_synthetic, f'global_hue_{hue_strength}'))
#     for index, image_path in tqdm(enumerate(images_list)):
#         img_id = os.path.split(image_path)[-1]
#         x = to_tensor(Image.open(image_path))
#         distort_img = global_hue_distort(x, hue_strength)
#         # already in 0 to one
#         if index == 0:
#             print(distort_img.mean(), distort_img.min(), distort_img.max())
        
#         img_out = TF.to_pil_image(torch.clip(distort_img, 0, 1))
#         img_out.save(os.path.join(base_synthetic, f'global_hue_{hue_strength}', f'{img_id}'))

# for function_name, function in [('dark', dark), ('bright', bright), ('zero', zero), ('noise', noise), ('blur', blur)]:
#     print(f"Processing local effect, {function_name}")
#     if not os.path.exists(os.path.join(base_synthetic, function_name)):
#         os.makedirs(os.path.join(base_synthetic, function_name))
#     for index, image_path in tqdm(enumerate(images_list)):
#         img_id = os.path.split(image_path)[-1]
#         x = to_tensor(Image.open(image_path))
#         mod_img = call_box_artifact(x, function=function)
#         # mod_img = normalize(mod_img)
#         if index == 0:
#             print(mod_img.mean(), mod_img.min(), mod_img.max())
        
#         img_out = TF.to_pil_image(torch.clip(mod_img, 0, 1))
#         img_out.save(os.path.join(base_synthetic, function_name, f'{img_id}'))



# for noise_level in [0, .001, .01,.05, .1, .2,.3, .4,.5]:
#     rstate = np.random.RandomState(seed)
#     noise = np.random.normal(loc=0.0,scale=noise_level,size=(256,256,3))
#     if not os.path.exists(os.path.join(base_synthetic, f'global_noise_{noise_level}')):
#         os.makedirs(os.path.join(base_synthetic, f'global_noise_{noise_level}'))
#     print("Processing noise, strength: ", noise_level)
#     for index, image_path in tqdm(enumerate(images_list)):
#         img_id = os.path.split(image_path)[-1]
#         x = np.array(Image.open(image_path),dtype=np.float32) / 255
#         x += noise 
#         x *= 255 
#         Image.fromarray(np.clip(x,0,255).astype(np.uint8)).save(os.path.join(base_synthetic, f'global_noise_{noise_level}', f'{img_id}'))

channel_opts =  [[0],[1],[2],[0,1],[0,2],[1,2]]
for function_name, function in [('color_patch_'+str(c), partial(color_patch,channels=c)) for c in channel_opts]:
    print(f"Processing local effect, {function_name}")
    if not os.path.exists(os.path.join(base_synthetic, function_name)):
        os.makedirs(os.path.join(base_synthetic, function_name))
    for index, image_path in tqdm(enumerate(images_list)):
        img_id = os.path.split(image_path)[-1]
        x = to_tensor(Image.open(image_path))
        mod_img = call_box_artifact(x, function=function)
        # mod_img = normalize(mod_img)
        if index == 0:
            print(mod_img.mean(), mod_img.min(), mod_img.max())
        
        img_out = TF.to_pil_image(torch.clip(mod_img, 0, 1))
        img_out.save(os.path.join(base_synthetic, function_name, f'{img_id}'))

# %%
