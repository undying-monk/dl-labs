import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize_img(image):
    img = image.permute(1, 2, 0).numpy()
    img = std * img + mean  # Un-normalize
    img = np.clip(img, 0, 1)  # Clip boundaries [3,256,256]
    return img

def plot_segmentation(image, target):
    img_1 = denormalize_img(image)
    img_1 = transforms.ToPILImage()(img_1)
    # img_1 = img_1.permute(1,2,0).cpu()

    _, (ax1,ax2, ax3) = plt.subplots(1,3,figsize=(8, 8))
    ax1.imshow(img_1)
    ax1.axis("off")


    mask_2d = target.squeeze().cpu().numpy() # [B, 1, 256,256] => [B,256,256]
    print("mask_2d", mask_2d)
    ax2.imshow(mask_2d, cmap="jet")
    ax2.axis("off")

    ax3.imshow(img_1)
    # alpha controls the transparency of the mask overlay
    ax3.imshow(mask_2d, cmap="jet", alpha=.5, interpolation="nearest")
    ax3.set_title("Overlay")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def plot_binary_segmentation(image, mask):
    img = denormalize_img(image)
    img = transforms.ToPILImage()(img)

    print("mask shape", mask.shape)
    mask = mask.squeeze(0).numpy() # [B, 1, 256,256] => [B,256,256]
    print("mask shape", mask.shape)

    # _, (ax1,ax2) = plt.subplots(1,2,figsize=(15, 5))
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.axis("off")

    plt.figure(figsize=(20,10))
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()