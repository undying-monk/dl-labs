import tensorflow as tf
import numpy as np
import torchvision.ops as ops
import torch
import pandas as pd
import math

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        print("anchors", anchors.split(','))
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def iou(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # calculate intersection
    xi1 = max(box1_x1,box2_x1)
    yi1 = max(box1_y1,box2_y1)
    xi2 = min(box1_x2,box2_x2)
    yi2 = min(box1_y2,box2_y2)
    inter_width = max(0,yi2 - yi1)
    inter_height = max(0,xi2 - xi1)
    inter_area = inter_width*inter_height

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ## (≈ 3 lines)
    box1_area = (box1_x2-box1_x1)*((box1_y2-box1_y1))
    box2_area = (box2_x2-box2_x1)*((box2_y2-box2_y1))
    union_area = box1_area + box2_area - inter_area
    iou = inter_area/union_area
    return iou

def iou_anchor(bb,anchor):
    inter = min(bb[0], anchor[0]) * min(bb[1], anchor[1])
    bb_area = bb[0] * bb[1]
    anchor_area = anchor[0] * anchor[1]
    union = bb_area + anchor_area - inter
    return inter/union


def find_highest_iou_anchor(bounding_box,anchors):
    best_anchor = None
    best_iou = -1

    for anchor_idx, anchor in enumerate(anchors):
        iou = iou_anchor(bounding_box, anchor)

        if iou > best_iou:
            best_iou = iou
            best_anchor = anchor_idx

    return best_anchor


def encode_yolo_target(image, target, anchors, grid_shape, num_classes):
    bounding_boxes = target["boxes"]
    labels = target["labels"]

    width, height = image.size
    grid_height = height/grid_shape[0]
    grid_width = width/grid_shape[1]

    # label_data = np.zeros((grid_shape[0],grid_shape[1], len(anchors), 5+num_classes))
    label_data = torch.full((grid_shape[0],grid_shape[1], len(anchors), 5+num_classes), 0.0, dtype=torch.float32)

    # print("label_data", label_data.shape, "image_size", image.size) 
    for i, box in enumerate(bounding_boxes):
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]

        # find central point to find grid cell
        midpoint_x = (xmax - xmin) /2
        midpoint_y = (ymax - ymin) /2
        b_w = (xmax - xmin) / grid_width # calculate width and convert into grid units
        b_h = (ymax - ymin) / grid_height # calculate width and convert into grid units

        grid_x = int(midpoint_x / grid_width)
        grid_y = int(midpoint_y / grid_height)

        t_x = (midpoint_x % grid_width) / grid_width # position inside cell
        t_y = (midpoint_y % grid_height) / grid_height

        # print("position",t_x,t_y, midpoint_x, midpoint_y, grid_height)

        # find match anchor box with label bounding box
        best_anchor = find_highest_iou_anchor([b_w,b_h], anchors)
        # print("bounding box", f"[{b_w:.2f}, {b_h:.2f}]", "- anchor", anchors[best_anchor])
        t_w = math.log(b_w/anchors[best_anchor][0] )
        t_h = math.log(b_h/anchors[best_anchor][1] )

        # position
        label_data[grid_x, grid_y, best_anchor, 0] = t_x 
        label_data[grid_x, grid_y, best_anchor, 1] = t_y
        # size
        label_data[grid_x, grid_y, best_anchor, 2] = t_w  # width
        label_data[grid_x, grid_y, best_anchor, 3] = t_h  # height

        # object exists
        label_data[grid_x, grid_y, best_anchor, 4] = 1.0

        # class
        label_data[grid_x, grid_y, best_anchor, 5 + labels[i]] = 1.0
        # print("label_data", label_data[grid_x, grid_y, best_anchor, :]) 
    return label_data

# convert [B,Channels, GRID, GRID] into [B, GRID, GRID, num_anchors, 5+num_classes]
def encode_archor(x, num_anchors, num_classes):
    B, C, H, W = x.shape

    x = x.permute(0, 2, 3, 1)
    # [B, 13, 13, 125]

    x = x.reshape(
        B,
        H,
        W,
        num_anchors,
        5 + num_classes
    )
    return x