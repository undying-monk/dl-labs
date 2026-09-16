from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torch
from object_detection.car.utils.util import get_sorted_iou_anchors
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
        self.anchors = anchors

    def __call__(self, target):
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
    
            best_anchor = get_sorted_iou_anchors(box, torch.tensor(self.anchors), occupied[grid_y, grid_x, :])
            if best_anchor == -1:
                print(
                    f"WARNING: no free anchor for "
                    f"box={box.tolist()}, "
                    f"class={int(labels[i])}, "
                    f"cell=({grid_y},{grid_x})"
                )
                continue
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