import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from utils.transform import Letterbox

# 1. Define standard image transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


data_transforms = v2.Compose([
    # transforms.Resize((416,416)),
    # transforms.RandomRotation(degrees=15),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomAutocontrast(0.1),
    v2.ToImage(),                          # 1. Converts PIL Image to a Tensor image
    # v2.Resize((256,256)),
    Letterbox((256, 256)),
    v2.ToDtype(torch.float32, scale=True), # 2. Converts to Float32 AND sca# les values to [0.0, 1.0]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Always last
])

def target_transform(mask_pil):
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
    
    # Shift Oxford-IIIT Pet labels: Trimap (1, 2, 3) -> Class Indices (0, 1, 2)
    # Class 0: Pet / Foreground (The animal's body)
    # Class 1: Background (Surrounding environment like grass, carpet, or walls)
    # Class 2: Contour / Border (The outline area where the animal meets the background)
    return mask_tensor - 1

def OxfordIIITPetTrainDataset():
    return datasets.OxfordIIITPet(
        root="./data",
        split="trainval",       # Supports "trainval" or "test"
        target_types="segmentation", # Options: "category", "binary-category", "segmentation"
        transform=data_transforms,
        target_transform=target_transform,
        download=True
    )

def OxfordIIITPetTestDataset():
    return datasets.OxfordIIITPet(
        root="./data",
        split="test",       # Supports "trainval" or "test"
        target_types="segmentation", # Options: "category", "binary-category", "segmentation"
        transform=data_transforms,
        target_transform=target_transform,
        download=True
    )

def detection_collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets