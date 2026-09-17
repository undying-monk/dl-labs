from torch.utils.data import Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET
import torch
from utils.util import get_sorted_iou_anchors
from torchvision.transforms import v2
from torchvision import tv_tensors

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


CLASS_TO_ID = {
    name: i
    for i, name in enumerate(VOC_CLASSES)
}

print(CLASS_TO_ID)

class VOCDataset(Dataset):
    def __init__(self, root, split_file, batch_size=64, grid_shape=(7, 7), anchors=[], transform=None, target_transform=None):
        self.root = root
        self.annotations_dir = self.root + '/' + "Annotations"
        self.img_dir =  self.root + '/' + "JPEGImages"
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.grid_shape = grid_shape
        self.num_anchors = len(anchors)
        self.num_classes = len(VOC_CLASSES)
        self.anchors = anchors

        with open(split_file, "r") as f:
            self.image_ids = [
                line.strip()
                for line in f
                if line.strip()
            ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path =self.img_dir + '/' + f"{image_id}.jpg"
        annotation_path =self.annotations_dir + '/' + f"{image_id}.xml"

        boxes, labels = self._parse_annotation(
            annotation_path
        )

        image = Image.open(
            img_path
        ).convert("RGB")

        width, height = image.size
        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=(height, width),
        )
 

        target = {
            'image_id': image_id,
            'labels': labels,
            'boxes': boxes,
        }
        # print("image",image_id, len(labels))
        # target = encode_yolo_target(image, target, self.anchors, self.grid_shape, self.num_classes)
        if self.transform:
            image, target = self.transform(
                image,
                target,
            )
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def _parse_annotation(self, path):

        tree = ET.parse(path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):

            class_name = obj.find(
                "name"
            ).text

            class_id = CLASS_TO_ID[
                class_name
            ]

            bndbox = obj.find("bndbox")

            xmin = float(
                bndbox.find("xmin").text
            )
            ymin = float(
                bndbox.find("ymin").text
            )
            xmax = float(
                bndbox.find("xmax").text
            )
            ymax = float(
                bndbox.find("ymax").text
            )

            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax,
            ])

            labels.append(class_id)

        return boxes, labels


def detection_collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


class Letterbox(v2.Transform):
    def __init__(self, size=(416, 416), fill=0):
        super().__init__()

        self.target_h = size[0]
        self.target_w = size[1]
        self.fill = fill

    def _transform(self, inpt, params):
        return inpt

    def forward(self, image, target):
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

        image, target = resize(image, target)

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

        image, target = pad(image, target)

        return image, target


def get_transform(train=False):
    transforms = [
        v2.ToImage(),
    ]

    if train:
        transforms.append(
            v2.RandomHorizontalFlip(p=0.5)
        )

    transforms.extend([
        Letterbox((416, 416)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    return v2.Compose(transforms)


def get_dataloader_kwargs(device, max_workers=4):
    """Return DataLoader settings tuned for the available hardware."""
    num_workers = min(max_workers, os.cpu_count() or 1)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": str(device).startswith("cuda"),
    }
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    return kwargs


class YoloV2GridTransform:
    def __init__(self, base_shape, default_value=0.0, anchors=[], grid_width=0, grid_height=0):
        """
        Args:
            base_shape (tuple): The structural shape you want your target to have (e.g., (3, 3), (10,)).
            default_value (float): The initial background value for the tensor canvas.
        """
        self.base_shape = base_shape
        self.default_value = default_value
        self.grid_height = grid_height
        self.grid_width = grid_width
        # Original implementation stored a Python list and recreated this tensor for every box.
        # Cache it once; __call__ only moves it if the target boxes use another device.
        self.anchors = torch.as_tensor(anchors, dtype=torch.float32).clone()

    def __call__(self, target):
        """Vectorized replacement for the original per-object target-encoding loop."""
        bounding_boxes = target["boxes"]
        device = bounding_boxes.device
        boxes = bounding_boxes.to(device=device, dtype=torch.float32)
        labels = torch.as_tensor(target["labels"], device=device, dtype=torch.long)
        image_id = target["image_id"]
        grid_rows, grid_columns = self.base_shape[:2]
        num_anchors = len(self.anchors)
        anchors = self.anchors.to(device=device)
        target_tensor = torch.full(
            self.base_shape,
            self.default_value,
            dtype=torch.float32,
            device=device,
        )
        if boxes.numel() == 0:
            return target_tensor

        # Original loop calculated these values one box at a time. Each expression below
        # operates on every box in the image at once: boxes has shape [N, 4].
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        midpoint_x = boxes[:, 0] + box_widths / 2
        midpoint_y = boxes[:, 1] + box_heights / 2
        box_wh = torch.stack((box_widths / self.grid_width, box_heights / self.grid_height), dim=1)

        grid_x = torch.div(midpoint_x, self.grid_width, rounding_mode="floor").long().clamp(0, grid_columns - 1)
        grid_y = torch.div(midpoint_y, self.grid_height, rounding_mode="floor").long().clamp(0, grid_rows - 1)
        t_x = midpoint_x / self.grid_width - grid_x
        t_y = midpoint_y / self.grid_height - grid_y

        # Compute every object/anchor IoU together: [N, 2] vs [A, 2] -> [N, A].
        # This replaces get_sorted_iou_anchors(...) inside the original object loop.
        intersection = torch.minimum(box_wh[:, None, :], anchors[None, :, :]).prod(dim=-1)
        box_areas = box_wh.prod(dim=-1, keepdim=True)
        anchor_areas = anchors.prod(dim=-1).unsqueeze(0)
        anchor_order = torch.argsort(
            intersection / (box_areas + anchor_areas - intersection + 1e-16),
            dim=1,
            descending=True,
        )

        assigned_anchors = torch.full((len(boxes),), -1, dtype=torch.long, device=device)
        occupied = torch.zeros(grid_rows * grid_columns * num_anchors, dtype=torch.bool, device=device)
        object_indices = torch.arange(len(boxes), device=device)

        # Assignment still needs collision resolution because only one object can use a
        # (grid cell, anchor) slot. This loop is bounded by A (five), not N objects.
        for rank in range(num_anchors):
            unresolved = object_indices[assigned_anchors == -1]
            if unresolved.numel() == 0:
                break
            candidate_anchors = anchor_order[unresolved, rank]
            slots = ((grid_y[unresolved] * grid_columns + grid_x[unresolved]) * num_anchors + candidate_anchors)
            available = ~occupied[slots]
            unresolved = unresolved[available]
            slots = slots[available]
            candidate_anchors = candidate_anchors[available]
            if unresolved.numel() == 0:
                continue

            first_object_for_slot = torch.full(
                (occupied.numel(),),
                len(boxes),
                dtype=torch.long,
                device=device,
            )
            first_object_for_slot.scatter_reduce_(0, slots, unresolved, reduce="amin", include_self=True)
            winners = unresolved == first_object_for_slot[slots]
            winner_indices = unresolved[winners]
            winner_slots = slots[winners]
            assigned_anchors[winner_indices] = candidate_anchors[winners]
            occupied[winner_slots] = True

        failed_indices = torch.where(assigned_anchors == -1)[0]
        if failed_indices.numel() > 0:
            failed_index = failed_indices[0]
            raise ValueError(
                f"Unable to encode object for image '{image_id}': all {num_anchors} anchors are occupied "
                f"in cell ({grid_y[failed_index].item()},{grid_x[failed_index].item()}) "
                f"for box={boxes[failed_index].tolist()}, class={labels[failed_index].item()}."
            )

        # Advanced indexing writes every encoded object in one operation, replacing the
        # original scalar target_tensor[grid_y, grid_x, best_anchor, ...] assignments.
        selected_anchors = anchors[assigned_anchors]
        target_tensor[grid_y, grid_x, assigned_anchors, 0] = t_x
        target_tensor[grid_y, grid_x, assigned_anchors, 1] = t_y
        target_tensor[grid_y, grid_x, assigned_anchors, 2] = torch.log(box_wh[:, 0] / selected_anchors[:, 0])
        target_tensor[grid_y, grid_x, assigned_anchors, 3] = torch.log(box_wh[:, 1] / selected_anchors[:, 1])
        target_tensor[grid_y, grid_x, assigned_anchors, 4] = 1.0
        target_tensor[grid_y, grid_x, assigned_anchors, 5 + labels] = 1.0
        return target_tensor

    def _encode_loop(self, target):
        """Original per-object implementation retained only as a reference; __call__ does not use it."""
        """
        Args:
            label_info (tuple/dict): A structure containing your target indices and values.
        """
        # 1. Initialize a blank canvas tensor with your custom shape
        target_tensor = torch.full(self.base_shape, self.default_value, dtype=torch.float32)
        
        # 2. Extract values from your dataset item
        bounding_boxes = target["boxes"]
        labels = target["labels"]
        image_id = target["image_id"]
        S = self.base_shape[0]
        A = len(self.anchors)
        occupied = torch.zeros(
            S,
            S,
            A,
            dtype=torch.bool,
            device=bounding_boxes.device,
        )
        
        # 3. Explicitly set values at the specified indices
        # If indices is a tuple of coordinates (e.g., (row_array, col_array)), 
        # PyTorch handles advanced multi-dimensional index assignment naturally.
        for i, box in enumerate(bounding_boxes):
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]

            # find central point to find grid cell
            midpoint_x = xmin + (xmax - xmin) /2
            midpoint_y = ymin + (ymax - ymin) /2
            b_w = (xmax - xmin) / self.grid_width # calculate width and convert into grid units
            b_h = (ymax - ymin) / self.grid_height # calculate width and convert into grid units

            # print("box", box, "midpoint_x", midpoint_x,"midpoint_y", midpoint_y, self.grid_width)

            grid_x = int(midpoint_x / self.grid_width)
            grid_y = int(midpoint_y / self.grid_height)
    
            t_x = (midpoint_x % self.grid_width) / self.grid_width # position inside cell
            t_y = (midpoint_y % self.grid_height) / self.grid_height
    
            box_wh = torch.stack((b_w, b_h))
            best_anchor = get_sorted_iou_anchors(
                box_wh,
                torch.tensor(self.anchors),
                occupied[grid_y, grid_x, :],
            )
            if best_anchor == -1:
                raise ValueError(
                    f"Unable to encode object for image '{image_id}': "
                    f"all {A} anchors are occupied in cell ({grid_y},{grid_x}) "
                    f"for box={box.tolist()}, class={int(labels[i])}."
                )
            occupied[grid_y, grid_x, best_anchor] = True

            # print("bounding box", f"[{b_w:.2f}, {b_h:.2f}]", "- anchor", anchors[best_anchor])
            t_w = torch.log(b_w/self.anchors[best_anchor][0] )
            t_h = torch.log(b_h/self.anchors[best_anchor][1] )
    
            # position
            target_tensor[grid_y, grid_x, best_anchor, 0] = t_x 
            target_tensor[grid_y, grid_x, best_anchor, 1] = t_y
            # size
            target_tensor[grid_y, grid_x, best_anchor, 2] = t_w  # width
            target_tensor[grid_y, grid_x, best_anchor, 3] = t_h  # height
    
            # object exists
            target_tensor[grid_y, grid_x, best_anchor, 4] = 1.0
    
            # class
            target_tensor[grid_y, grid_x, best_anchor, 5 + labels[i]] = 1.0

        # encoded_count = (
        #     target_tensor[..., 4] == 1
        # ).sum().item()
        # gt_count = len(labels)

        # print(
        #     f"image {image_id}: "
        #     f"GT={gt_count}, "
        #     f"diff={gt_count != encoded_count}, "
        #     f"encoded={encoded_count}"
        # )
        return target_tensor
