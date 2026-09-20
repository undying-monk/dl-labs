import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# 1. Define standard image transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

class Letterbox(v2.Transform):
    def __init__(self, size=(256, 256), fill=0):
        super().__init__()

        self.target_h = size[0]
        self.target_w = size[1]
        self.fill = fill

    def _transform(self, inpt, params):
        return inpt

    def forward(self, image):
        # image is expected to be a TVTensor Image
        # target["boxes"] is a BoundingBoxes TVTensor

        _, h, w = image.shape

        target_h = self.target_h
        target_w = self.target_w

        # ----------------------------------------
        # 1. Calculate scale
        # ----------------------------------------

        scale = min(
            target_w / w,
            target_h / h,
        )

        new_w = round(w * scale)
        new_h = round(h * scale)

        # ----------------------------------------
        # 2. Resize image + boxes together
        # ----------------------------------------

        resize = v2.Resize(
            size=(new_h, new_w)
        )

        image = resize(image)

        # ----------------------------------------
        # 3. Calculate padding
        # ----------------------------------------
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

        pad_right = target_w - new_w - pad_left
        pad_bottom = target_h - new_h - pad_top

        # torchvision Pad order:
        # [left, top, right, bottom]

        pad = v2.Pad(
            padding=[
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ],
            fill=self.fill,
        )

        image = pad(image)

        return image

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