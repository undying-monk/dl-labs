import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize_img(image):
    img = image.permute(1, 2, 0).numpy()
    img = std * img + mean  # Un-normalize
    img = np.clip(img, 0, 1)  # Clip boundaries [3,256,256]
    return transforms.ToPILImage()(img)

def plot_segmentation(image, target):
    img_1 = denormalize_img(image)
    # img_1 = img_1.permute(1,2,0).cpu()

    _, (ax1,ax2, ax3) = plt.subplots(1,3,figsize=(8, 8))
    ax1.imshow(img_1)
    ax1.axis("off")

    mask_2d = target.squeeze().cpu().numpy() # [B, 1, 256,256] => [B,256,256]
    ax2.imshow(mask_2d, cmap="jet")
    ax2.axis("off")

    ax3.imshow(img_1)
    # alpha controls the transparency of the mask overlay
    ax3.imshow(mask_2d, cmap="jet", alpha=.5, interpolation="nearest")
    ax3.set_title("Overlay")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()