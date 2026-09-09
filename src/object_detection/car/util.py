import tensorflow as tf
import numpy as np
import torchvision.ops as ops
import torch
import pandas as pd
import math
from PIL import Image, ImageDraw, ImageFont
import colorsys
import random
from torchvision.ops import nms

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


def decode_predictions(
    prediction,
    anchors,
    stride=32,
    conf_threshold=0.3,
):
    """
    prediction:
        [13, 13, 5, 25]

    anchors:
        [(aw, ah), ...]

    returns:
        boxes  [N, 4]
        scores [N]
        labels [N]
    """

    S = prediction.shape[0]
    num_anchors = prediction.shape[2]

    boxes = []
    scores = []
    labels = []

    for gy in range(S):
        for gx in range(S):
            for anchor_idx in range(num_anchors):

                pred = prediction[
                    gy, gx, anchor_idx
                ]

                tx, ty, tw, th = pred[:4]

                # --------------------------------
                # Decode center
                # --------------------------------



                cx = (
                    torch.sigmoid(tx) + gx
                ) * stride

                cy = (
                    torch.sigmoid(ty) + gy
                ) * stride

                # --------------------------------
                # Decode size
                # --------------------------------

                anchor_w, anchor_h = anchors[anchor_idx]

                w = (
                    torch.exp(tw)
                    * anchor_w
                    * stride
                )

                h = (
                    torch.exp(th)
                    * anchor_h
                    * stride
                )

                # --------------------------------
                # xywh -> xyxy
                # --------------------------------

                xmin = cx - w / 2
                ymin = cy - h / 2
                xmax = cx + w / 2
                ymax = cy + h / 2

                box = torch.stack([
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ])

                # --------------------------------
                # Objectness
                # --------------------------------

                objectness = torch.sigmoid(
                    pred[4]
                )
                

                # --------------------------------
                # Classes
                # --------------------------------

                class_probs = torch.sigmoid(
                    pred[5:]
                )


                class_prob, class_id = torch.max(
                    class_probs,
                    dim=0
                )

                # --------------------------------
                # Final confidence
                # --------------------------------

                confidence = (
                    objectness * class_prob
                )

                if confidence >= conf_threshold:
                    boxes.append(box)
                    scores.append(confidence)
                    labels.append(class_id)

    if len(boxes) == 0:
        return (
            torch.empty((0, 4)),
            torch.empty((0,)),
            torch.empty((0,), dtype=torch.long),
        )

    return (
        torch.stack(boxes),
        torch.stack(scores),
        torch.stack(labels),
    )

def apply_nms(
    boxes,
    scores,
    labels,
    iou_threshold=0.5,
):
    keep = nms(
        boxes,
        scores,
        iou_threshold,
    )

    return (
        boxes[keep],
        scores[keep],
        labels[keep],
    )

def denormalize(image):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=image.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=image.device,
    ).view(3, 1, 1)

    return image * std + mean