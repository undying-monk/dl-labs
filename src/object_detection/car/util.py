import tensorflow as tf
import numpy as np
import torchvision.ops as ops
import torch
import pandas as pd
import math

# filter scores box over threshold
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold):
    # Arguments:
    # boxes -- tensor of shape (19, 19, 5, 4)
    # box_confidence -- tensor of shape (19, 19, 5, 1)
    # box_class_probs -- tensor of shape (19, 19, 5, 80)
    # threshold -- real value, if [ highest class probability score < threshold],
    #              then get rid of the corresponding box


    # Returns:
    # scores -- tensor of shape (None,), containing the class probability score for selected boxes
    # boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    # classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    scores = box_class_probs * box_confidence # (19, 19, 5, 80)
    print(scores.shape)
    # retrieve index with max score per row
    box_classes = tf.math.argmax(scores, axis=-1) # (19, 19, 5, 1) keepdims=False => (19, 19, 5)
    print("box_classes",box_classes.shape)
    # retrieve max score per row
    box_classes_score = tf.math.reduce_max(scores, axis=-1) # (19, 19, 5) keepdims=False => (19, 19, 5)
    print("box_classes_score",box_classes_score.shape)

    filtering = (box_classes_score > threshold)
    print("filter",filtering.shape) # (19, 19, 5)

    boxes = tf.boolean_mask(boxes, filtering) # (19, 19, 5, 4)
    classes = tf.boolean_mask(box_classes, filtering) # (19, 19, 5)
    scores = tf.boolean_mask(box_classes_score, filtering) # (19, 19, 5)

    return scores, boxes, classes

# pick highest score box and remove other overlap box with iou_threshold (iou >= iou_threshold)
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10,iou_threshold=0.5):
    max_boxes_tensor = tf.Variable(max_boxes,dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor,iou_threshold)
    return tf.gather(scores,nms_indices),  tf.gather(boxes,nms_indices), tf.gather(classes,nms_indices)

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0] * 1.0
    width = image_shape[1] * 1.0
    image_dims = tf.keras.backend.stack([height, width, height, width])
    image_dims = tf.keras.backend.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    
    # Retrieve outputs of the YOLO model (≈1 line)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)
    
    # Scale boxes back to original image shape (720, 1280 or whatever)
    boxes = scale_boxes(boxes, image_shape) # Network was trained to run on 608x608 images

    # Use one of the functions you've implemented to perform Non-max suppression with 
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    # YOUR CODE STARTS HERE
    
    
    # YOUR CODE ENDS HERE
    
    return scores, boxes, classes


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