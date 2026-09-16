import tensorflow as tf
import numpy as np
import torchvision.ops as ops
import torch
import pandas as pd
import math
from PIL import Image, ImageDraw, ImageFont
import colorsys
import random


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
    print("filter",filtering.shape, boxes.shape) # (19, 19, 5)

    boxes = tf.boolean_mask(boxes, filtering) # (19, 19, 5, 4)
    classes = tf.boolean_mask(box_classes, filtering) # (19, 19, 5)
    scores = tf.boolean_mask(box_classes_score, filtering) # (19, 19, 5)
    return scores, boxes, classes

# pick highest score box and remove other overlap box with iou_threshold (iou >= iou_threshold)
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10,iou_threshold=0.5):
    max_boxes_tensor = tf.Variable(max_boxes,dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor,iou_threshold)
    return tf.gather(scores,nms_indices),  tf.gather(boxes,nms_indices), tf.gather(classes,nms_indices)


# calculate box ymin,xmin,ymax,xmax by box center point x_y and box w_h
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
    print("boxes.shape", boxes.shape, "image_dims", image_dims.shape)
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

   # converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of `yolo_filter_boxes`
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


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    #image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score = scores.numpy()[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # # My kingdom for a good redistributable image drawing library.
        # for i in range(thickness):
        #     draw.rectangle(
        #         [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
                        [left, top, right, bottom], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)

def preprocess_image(img_path, model_image_size):
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data