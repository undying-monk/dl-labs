import numpy as np
import torchvision.ops as ops
import torch
import math
from PIL import Image, ImageDraw
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.ops import batched_nms

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
    return inter / union

def iou_anchors(bb,anchors):
    inter = torch.minimum(bb[0], anchors[:, 0]) * torch.minimum(bb[1], anchors[:,1])
    bb_area = bb[0] * bb[1]
    anchor_area = anchors[:, 0] * anchors[:, 1]
    union = bb_area + anchor_area - inter
    return inter / (union + 1e-16)

def get_sorted_iou_anchors(box, anchors, occupied):
    anchor_ious = iou_anchors(box, anchors)
    sorted_anchor_idx = torch.argsort(anchor_ious,descending=True)
    best_anchor = -1
    for anchor_idx_tensor in sorted_anchor_idx:
        anchor_idx = int(anchor_idx_tensor.item())
        if not occupied[anchor_idx]:
            best_anchor = anchor_idx
            break
    return best_anchor

def find_highest_iou_anchor(bounding_box,anchors):
    best_anchor = None
    best_iou = -1

    for anchor_idx, anchor in enumerate(anchors):
        iou = iou_anchor(bounding_box, anchor)

        if iou > best_iou:
            best_iou = iou
            best_anchor = anchor_idx

    return best_anchor, best_iou

def encode_yolo_target(target, anchors, grid_width, grid_height, grid_shape, num_classes):
    bounding_boxes = target["boxes"]
    labels = target["labels"]

    # width, height = image.size
    # grid_height = height/grid_shape[0]
    # grid_width = width/grid_shape[1]

    # label_data = np.zeros((grid_shape[0],grid_shape[1], len(anchors), 5+num_classes))
    label_data = torch.full((*grid_shape, len(bounding_boxes), 5+num_classes), 0.0, dtype=torch.float32)
    S = grid_shape[0]
    A = len(anchors)
    occupied = torch.zeros(
        S,
        S,
        A,
        dtype=torch.bool,
        device=bounding_boxes.device,
    )
    # print("label_data", label_data.shape, "image_size", image.size) 
    for i, box in enumerate(bounding_boxes):
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]

        # find central point to find grid cell
        midpoint_x = (xmin + xmax) /2
        midpoint_y = (ymin + ymax) /2
        b_w = (xmax - xmin) / grid_width
        b_h = (ymax - ymin) / grid_height

        # grid_x = int(midpoint_x / grid_width)
        # grid_y = int(midpoint_y / grid_height)

        g_x = midpoint_x / grid_width
        g_y = midpoint_y / grid_height

        grid_x = int(torch.floor(g_x).item())
        grid_y = int(torch.floor(g_y).item())
        
        # Clamp against boundary case where cx/cy == image_size
        grid_x = min(max(grid_x, 0), S - 1)
        grid_y = min(max(grid_y, 0), S - 1)

        t_x = g_x - grid_x # position inside cell
        t_y = g_y - grid_y # position inside cell
        best_anchor = get_sorted_iou_anchors(box, anchors, occupied[grid_y, grid_x, :])
        if best_anchor == -1:
            print(
                f"WARNING: no free anchor for "
                f"box={box.tolist()}, "
                f"class={int(labels[i])}, "
                f"cell=({grid_y},{grid_x})"
            )
            continue

        # print("label index", i, VOC_CLASSES[labels[i]], grid_x, grid_y, "best_anchor", best_anchor)
        occupied[grid_y, grid_x, best_anchor] = True

        # print("bounding box", f"[{b_w:.2f}, {b_h:.2f}]", "- anchor", anchors[best_anchor])
        t_w = torch.log(b_w/anchors[best_anchor][0] )
        t_h = torch.log(b_h/anchors[best_anchor][1] )
        print("encode",t_x.item(), t_y.item(), t_w.item(), t_h.item())

        # position
        label_data[grid_y, grid_x, best_anchor, 0] = t_x 
        label_data[grid_y, grid_x, best_anchor, 1] = t_y
        
        # size
        label_data[grid_y, grid_x, best_anchor, 2] = t_w  # width
        label_data[grid_y, grid_x, best_anchor, 3] = t_h  # height

        # object exists
        label_data[grid_y, grid_x, best_anchor, 4] = 1.0

        # class
        label_data[grid_y, grid_x, best_anchor, 5 + labels[i]] = 1.0
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


def preprocess_image(img_path, model_image_size):
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data


def decode_batch_predictions(
    predictions,
    anchors,
    stride=32,
    objectness_threshold=0.01,
    conf_threshold=0.05,
    nms_threshold=0.5,
):
    """
    predictions: [B, S, S, A, 5+C]
    """

    B, S, _, A, D = predictions.shape
    C = D - 5

    device = predictions.device
    dtype = predictions.dtype
    anchors = torch.from_numpy(anchors).to(device)

    # ---------------------------------------------
    # Objectness
    # ---------------------------------------------

    objectness = torch.sigmoid(predictions[..., 4])
    # [B, S, S, A]

    # Filter obvious background predictions
    obj_mask = objectness > objectness_threshold

    results = []

    # We still loop over batch because TorchMetrics
    # expects one prediction dict per image.
    for b in range(B):
        mask = obj_mask[b]

        if not mask.any():
            results.append({
                "boxes": torch.empty(
                    (0, 4),
                    device=device,
                    dtype=dtype,
                ),
                "scores": torch.empty(
                    (0,),
                    device=device,
                    dtype=dtype,
                ),
                "labels": torch.empty(
                    (0,),
                    device=device,
                    dtype=torch.long,
                ),
            })
            continue

        # ---------------------------------------------
        # Select only promising predictions
        # ---------------------------------------------

        pred = predictions[b][mask]

        # pred shape:
        # [N, 5+C]

        objectness_b = objectness[b][mask]

        # ---------------------------------------------
        # Classes
        # ---------------------------------------------

        class_probs = torch.sigmoid(
            pred[:, 5:]
        )

        class_prob, labels = class_probs.max(dim=1)

        scores = objectness_b * class_prob

        # ---------------------------------------------
        # Confidence filtering
        # ---------------------------------------------

        conf_mask = scores > conf_threshold

        pred = pred[conf_mask]
        objectness_b = objectness_b[conf_mask]
        class_prob = class_prob[conf_mask]
        labels = labels[conf_mask]
        scores = scores[conf_mask]

        if pred.numel() == 0:
            results.append({
                "boxes": torch.empty(
                    (0, 4),
                    device=device,
                    dtype=dtype,
                ),
                "scores": torch.empty(
                    (0,),
                    device=device,
                    dtype=dtype,
                ),
                "labels": torch.empty(
                    (0,),
                    device=device,
                    dtype=torch.long,
                ),
            })
            continue

        # ---------------------------------------------
        # Need grid indices for selected predictions
        # ---------------------------------------------

        # Recreate corresponding positions
        grid_y, grid_x, anchor_idx = torch.where(
            mask
        )

        # Apply confidence mask
        grid_y = grid_y[conf_mask]
        grid_x = grid_x[conf_mask]
        anchor_idx = anchor_idx[conf_mask]

        # ---------------------------------------------
        # Decode
        # ---------------------------------------------

        tx = pred[:, 0]
        ty = pred[:, 1]
        tw = pred[:, 2]
        th = pred[:, 3]
       
        anchor_wh = anchors[anchor_idx]

        aw = anchor_wh[:, 0]
        ah = anchor_wh[:, 1]

        cx = (
            tx
            + grid_x.to(dtype)
        ) * stride

        cy = (
            ty
            + grid_y.to(dtype)
        ) * stride

        w = (
            torch.exp(tw)
            * aw
            * stride
        )

        h = (
            torch.exp(th)
            * ah
            * stride
        )

        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2

        boxes = torch.stack(
            [xmin, ymin, xmax, ymax],
            dim=1,
        )

        # ---------------------------------------------
        # NMS
        # ---------------------------------------------

        keep = batched_nms(
            boxes,
            scores,
            labels,
            nms_threshold,
        )

        results.append({
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        })

    return results


def decode_predictions(
    prediction,
    anchors,
    stride=32,
    conf_threshold=.3,
    iou_threshold=.5,
    enable_nms=False,
):
    """
    prediction:
        [ 13, 13, 5, 25]

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
                
                # no object found
                if pred[4] == 0:
                    continue

                tx, ty, tw, th = pred[:4]

                # --------------------------------
                # Decode center
                # --------------------------------

                cx = (tx + gx) * stride
                cy = (ty + gy) * stride

                print("decode", tx.item(), ty.item(), tw.item(), th.item())


                # --------------------------------
                # Decode size
                # --------------------------------

                anchor_w, anchor_h = anchors[anchor_idx]

                # tw = log(bw/anchors[best_anchor])
                # bw = anchors[best_anchor] * e^(tw) * grid_width

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

                xmin = cx - (w / 2)
                ymin = cy - (h / 2)
                xmax = cx + (w / 2)
                ymax = cy + (h / 2)

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
    if enable_nms:
        return apply_nms(
            torch.stack(boxes),
            torch.stack(scores),
            torch.stack(labels),
            iou_threshold,
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
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
    ).view(3, 1, 1)

    return image * std + mean

def plot_bounding_boxes(image, boxes, labels, enable_grid=False):
    if type(boxes) == "list":
        boxes = torch.tensor(boxes)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(to_pil_image(draw_bounding_boxes(
        image,
        boxes,
        labels=labels,
        width=2,
    )))
    if enable_grid:
        show_yolo_grid(image, ax)
    plt.axis("off")
    plt.show()

def plot_anchors(anchors):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw rectangle shape width, height = 1,1
    # ax.add_patch(
    #     Rectangle(
    #         (0, 0),
    #         1, # width
    #         1, # height
    #         fill=False,
    #         linewidth=2,
    #     )
    # )

    # Anchor center
    cx = 0.5
    cy = 0.5

    ax.plot(cx, cy, marker="o")

    # Draw anchors shape
    for i, (aw, ah) in enumerate(anchors):
        xmin = cx - aw / 2 # in the left
        ymin = cy - ah / 2  # in the right

        rect = Rectangle(
            (xmin, ymin),
            aw,
            ah,
            fill=False,
            linewidth=1.5,
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymin,
            f"A{i}: {aw:.2f}×{ah:.2f}",
            fontsize=9,
        )

    # Show one grid cell from 0 to 1
    ax.set_xlim(-9, 10)
    ax.set_ylim(-7, 7)

    ax.set_aspect("equal")

    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    plt.show()


def show_yolo_grid(image, ax, grid_size=13):
    """
    image: torch.Tensor [C, H, W]
    """
    # Convert [C,H,W] -> [H,W,C]
    image_np = image.permute(1, 2, 0).cpu().numpy()

    H, W = image.shape[-2:]
    stride_x = W / grid_size
    stride_y = H / grid_size

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(image_np)

    # Vertical lines
    for i in range(grid_size + 1):
        x = i * stride_x
        ax.axvline(x=x, linewidth=0.8)

    # Horizontal lines
    for i in range(grid_size + 1):
        y = i * stride_y
        ax.axhline(y=y, linewidth=0.8)

    # Show grid coordinates
    for gy in range(grid_size):
        for gx in range(grid_size):
            x = (gx + 0.5) * stride_x
            y = (gy + 0.5) * stride_y

            ax.text(
                x,
                y,
                f"{gx},{gy}",
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.show()