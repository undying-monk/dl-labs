from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch
import os
from PIL import Image
import numpy as np
# from utils.transform import Letterbox

data_transforms = v2.Compose([
    v2.ToImage(),                          # 1. Converts PIL Image to a Tensor image
    v2.Resize(size=(256, 256), antialias=True),
    # Letterbox((256, 256)),
    v2.ToDtype(torch.float32, scale=True), # 2. Converts to Float32 AND sca# les values to [0.0, 1.0]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Always last
])

def target_transform(mask_pil):
    print("mask_pil", mask_pil.shape)

    # Standardize the PIL mask into a V2 wrapper
    mask_v2 = v2.functional.to_image(mask_pil)
    
    # CRITICAL: Force Nearest Neighbor interpolation so class labels (1, 2, 3) don't blur!
    mask_resized = v2.functional.resize(
        mask_v2, 
        size=(256, 256), 
        interpolation=v2.InterpolationMode.NEAREST
    )
    
    # Convert to standard PyTorch Long Tensor (required for CrossEntropyLoss)
    mask_tensor = mask_resized.long()
    return mask_tensor

class BinarySegmentationDataset(Dataset):
    def __init__(self, root, transform=data_transforms, target_transform=None):
        self.root = root
        self.mask_dir = self.root + '/' + "label_images_semantic"
        self.img_dir =  self.root + '/' + "original_images"
        self.transform = transform
        self.target_transform = target_transform

        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        mask = tv_tensors.Mask(Image.open(mask_path).convert("L")) # grayscale
        print("begin", mask)

        if self.transform:
            print("shape", image.shape)
            image, mask = self.transform(
                image,
                mask,
            )
           
        if self.target_transform:
            mask = self.target_transform(mask)

        print("mask after transform", mask)

        mask = (mask > 0.5).float() 
        print("mask result", mask)
        return image, mask