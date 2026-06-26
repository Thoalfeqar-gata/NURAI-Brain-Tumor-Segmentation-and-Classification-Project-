import torch
import torchvision
import cv2, numpy as np
from torch.utils.data import Dataset
from PIL import Image


class PretrainingDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, geom_transform=None, photo_transform=None, base_transform=None):
        super(PretrainingDataset, self).__init__(root=root)
        self.geom_transform = geom_transform
        self.photo_transform = photo_transform
        self.base_transform = base_transform # For ToTensor and Normalization

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        
        # 1. Apply geometric transforms to get the clean PIL image (spatially aligned target)
        if self.geom_transform is not None:
            clean_pil = self.geom_transform(img)
        else:
            clean_pil = img
            
        # 2. Apply photometric corruptions to get the augmented PIL image (model input)
        if self.photo_transform is not None:
            augmented_pil = self.photo_transform(clean_pil)
        else:
            augmented_pil = clean_pil
            
        # 3. Convert both PIL images to normalized PyTorch tensors
        if self.base_transform is not None:
            augmented_img = self.base_transform(augmented_pil)
            clean_img = self.base_transform(clean_pil)
        else:
            augmented_img = augmented_pil
            clean_img = clean_pil
            
        return augmented_img, clean_img, label


class CLAHETransform:
    def __init__(self, clipLimit=2.0, tileGridSize=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, img):
        # img: PIL Image (RGB) or grayscale
        arr = np.array(img)
        if arr.ndim == 2:  # grayscale
            out = self.clahe.apply(arr)
        else:
            # arr is RGB (HxWx3). Convert to LAB, apply CLAHE to L channel.
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(out)

