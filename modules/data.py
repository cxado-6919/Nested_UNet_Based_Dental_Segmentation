import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

EXPECTED_MAPPING = {
    0: 0,   15: 1,   29: 2,   38: 3,   53: 4,   67: 5,   75: 6,   76: 7,
    79: 8,  90: 9,   91: 10, 104: 11, 105: 12, 113: 13, 128: 14, 142: 15,
  150: 16, 151: 17,  164: 18, 166: 19, 179: 20, 188: 21, 192: 22, 202: 23,
  205: 24, 223: 25,  226: 26, 251: 27, 255: 28, 240: 29, 250: 30, 245: 31
}

def map_mask_fixed(mask):
    mapped = np.zeros_like(mask, dtype=np.int64)
    for orig_val, target in EXPECTED_MAPPING.items():
        mapped[mask == orig_val] = target
    unique_vals = np.unique(mask)
    for val in unique_vals:
        if val not in EXPECTED_MAPPING:
            raise ValueError(f"Mask contains unexpected value {val}. Expected keys: {list(EXPECTED_MAPPING.keys())}")
        return mapped

# albumentations 기반 데이터 증강 및 전처리
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.Normalize(mean=[0.3439, 0.3439, 0.3439], std=[0.2127, 0.2127, 0.2127]),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

val_test_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.3439, 0.3439, 0.3439], std=[0.2127, 0.2127, 0.2127]),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# X-ray 이미지 및 마스크 Dataset 클래스
class XRayDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_ids = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)
        mask_id = os.path.splitext(image_id)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_id)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].numpy()
        mask = torch.as_tensor(map_mask_fixed(mask), dtype=torch.long)
        return image, mask