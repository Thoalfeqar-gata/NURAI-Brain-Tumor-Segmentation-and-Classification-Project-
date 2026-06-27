import torch
import torch.nn as nn
import torchvision
import cv2, numpy as np, os
from torch.utils.data import Dataset
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import v2


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


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir = os.path.join('Data', 'Segmentation'), transform = None, train = True, seed = 42):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.train = train
        
        class_translation = {
            'Glioma' : 0,
            'Meningioma' : 1,
            'notumor' : 2,
            'Pituitary tumor' : 3
        }
        
        # collect image paths, mask paths, and class ids
        self.image_paths = []
        self.mask_paths = []
        self.class_ids = []
        if os.path.exists(self.dataset_dir):
            for class_name in os.listdir(self.dataset_dir):
                class_dir = os.path.join(self.dataset_dir, class_name)
                if not os.path.isdir(class_dir) or class_name not in class_translation:
                    continue
                for image_name in os.listdir(class_dir):
                    if 'mask' not in image_name and not image_name.startswith('.'):
                        filename, filetype = os.path.splitext(image_name)
                        maskname = filename + '_mask' + filetype
                        mask_path = os.path.join(class_dir, maskname)
                        if os.path.exists(mask_path):
                            self.image_paths.append(os.path.join(class_dir, image_name))
                            self.mask_paths.append(mask_path)
                            self.class_ids.append(class_translation[class_name])
        
        # split in to training and testing
        rng = np.random.default_rng(seed = seed)
        indices = np.arange(len(self.class_ids))
        rng.shuffle(indices)
        
        split_index = int(0.7 * len(indices))
        
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        
        if train:
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.mask_paths = [self.mask_paths[i] for i in train_indices]
            self.class_ids = [self.class_ids[i] for i in train_indices]
        else:
            self.image_paths = [self.image_paths[i] for i in test_indices]
            self.mask_paths = [self.mask_paths[i] for i in test_indices]
            self.class_ids = [self.class_ids[i] for i in test_indices]
            

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        image = torchvision.io.decode_image(self.image_paths[index], mode = torchvision.io.ImageReadMode.RGB)
        mask = torchvision.io.decode_image(self.mask_paths[index], mode = torchvision.io.ImageReadMode.GRAY)
        class_id = self.class_ids[index]

        # Apply rotation and horizontal flipping augmentation to both image and mask if training
        if self.train:
            # Horizontal flipping
            if torch.rand(1).item() > 0.5:
                image = torchvision.transforms.functional.hflip(image)
                mask = torchvision.transforms.functional.hflip(mask)
            
            # Rotation
            if torch.rand(1).item() > 0.5:
                angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())
                image = torchvision.transforms.functional.rotate(image, angle)
                mask = torchvision.transforms.functional.rotate(mask, angle)

        # Convert mask to float and scale it to [0.0, 1.0] by thresholding at 127
        mask = (mask > 127).to(dtype=torch.float32)

        # Apply transform to image
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.to(dtype=torch.float32) / 255.0

        image = v2.functional.resize(image, size = (224, 224), interpolation = v2.functional.InterpolationMode.BILINEAR)
        mask = v2.functional.resize(mask, size = (224, 224), interpolation = v2.functional.InterpolationMode.NEAREST)
        return image, mask, class_id


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. Apply sigmoid to convert raw logits to probabilities (0 to 1)
        probs = torch.sigmoid(logits)
        
        # 2. Flatten both tensors to 1D arrays for easy mathematical operations
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # 3. Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        # 4. Compute the Dice score using the smoothing factor
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 5. Return the loss (1 - score), as PyTorch optimizers minimize values
        return 1. - dice_score
    

def calculate_iou(pred_logits, true_mask, threshold=0.5, smooth=1e-5):
    # 1. Apply sigmoid and cast to float inside the function
    probs = torch.sigmoid(pred_logits)
    pred_binary = (probs > threshold).float()
    
    # 2. Flatten
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    true_flat = true_mask.view(true_mask.size(0), -1)
    
    # 3. Math
    intersection = (pred_flat * true_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def calculate_dice(pred_logits, true_mask, threshold=0.5, smooth=1e-5):
    # 1. Apply sigmoid and cast to float inside the function
    probs = torch.sigmoid(pred_logits)
    pred_binary = (probs > threshold).float()
    
    # 2. Flatten
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    true_flat = true_mask.view(true_mask.size(0), -1)
    
    # 3. Math
    intersection = (pred_flat * true_flat).sum(dim=1)
    predicted_sum = pred_flat.sum(dim=1)
    true_sum = true_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (predicted_sum + true_sum + smooth)
    return dice.mean().item()


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

