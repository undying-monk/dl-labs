from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import math
import numpy as np
from util import encode_yolo_target

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

    def __init_labels__(self):
        # if idx +1 * self.batch_size > self.__len__():
        #     batch_size = self.__len__() % batch_size
        # else:
        #     batch_size = self.batch_size

        # batch_size presents 3D num of rows, 7x7, num_anchors + p_obj + 4_box_coordianates + num_classes => 7x7x(5+5+3) = 7x7x8
        label_data = np.zeros((self.__len__(),*self.grid_shape, self.num_anchors, self.num_classes))
        return label_data


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path =self.img_dir + '/' + f"{image_id}.jpg"
        annotation_path =self.annotations_dir + '/' + f"{image_id}.xml"
        image = Image.open(
            img_path
        ).convert("RGB")

        boxes, labels = self._parse_annotation(
            annotation_path
        )

        target = {
            'image_id': image_id,
            'labels': labels,
            'boxes': boxes,
        }
        target = encode_yolo_target(image, target, self.anchors, self.grid_shape, self.num_classes)
        if self.transform:
            image = self.transform(image)
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


# def detection_collate_fn(batch):
#     images = []
#     targets = []

#     for image, target in batch:
#         images.append(image)
#         targets.append(target)

#     images = torch.stack(images)

#     return images, targets