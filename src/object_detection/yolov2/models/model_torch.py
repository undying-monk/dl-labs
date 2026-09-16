import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
)
from torchmetrics.detection import MeanAveragePrecision
from utils.util import encode_archor, decode_batch_predictions
from torchvision.ops import box_iou


class ConvBlock(nn.Module):
    """Standard Convolution -> Batch Normalization -> Leaky ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding: int | None = None):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Passthrough(nn.Module):
    """Reorganizes a 26x26x512 feature map into a 13x13x2048 map"""
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        

        B, C, H, W = x.size()
        # Reshape to fold spatial dimensions into channels
        x = x.view(B, C, H // self.stride, self.stride, W // self.stride, self.stride)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(B, C * self.stride * self.stride, H // self.stride, W // self.stride)

class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBlock(3, 32, 3, 1),
            nn.MaxPool2d(2, 2),
        )

        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
        )

        self.stage3 = nn.Sequential(
            ConvBlock(64, 128, 3, 1),
            ConvBlock(128, 64, 1, 1),
            ConvBlock(64, 128, 3, 1),
            nn.MaxPool2d(2, 2),
        )

        self.stage4 = nn.Sequential(
            ConvBlock(128, 256, 3, 1),
            ConvBlock(256, 128, 1, 1),
            ConvBlock(128, 256, 3, 1),
            nn.MaxPool2d(2, 2),
        )

        self.stage5 = nn.Sequential(
            ConvBlock(256, 512, 3, 1),
            ConvBlock(512, 256, 1, 1),
            ConvBlock(256, 512, 3, 1),
            ConvBlock(512, 256, 1, 1),
            ConvBlock(256, 512, 3, 1),
        )

        self.pool5 = nn.MaxPool2d(2, 2)

        self.stage6 = nn.Sequential(
            ConvBlock(512, 1024, 3, 1),
            ConvBlock(1024, 512, 1, 1),
            ConvBlock(512, 1024, 3, 1),
            ConvBlock(1024, 512, 1, 1),
            ConvBlock(512, 1024, 3, 1),
        )

    def forward(self, x):
        x = self.stage1(x)
        # print("stage1:", x.shape)

        x = self.stage2(x)
        # print("stage2:", x.shape)

        x = self.stage3(x)
        # print("stage3:", x.shape)

        x = self.stage4(x)
        # print("stage4:", x.shape)

        x = self.stage5(x)
        # print("stage5:", x.shape)

        x_stage5 = x.clone() 

        x = self.pool5(x)
        # print("pool5:", x.shape)

        x = self.stage6(x)
        # print("stage6:", x.shape)

        x_stage6 = x.clone()
        return x_stage5, x_stage6

class YOLOv2(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        
        self.darknet = Darknet19()                     #(1)
        self.route_conv = ConvBlock(512,64,kernel_size=1, stride=1)
        self.passthrough = Passthrough(2)              #(2)
        self.stage6 = nn.ModuleList([                  #(3)
            ConvBlock(1024, 1024, 3, 1), # 13x13
            ConvBlock(1024, 1024, 3, 1), # 13x13
        ])
        self.stage7 = nn.ModuleList([
            ConvBlock(1280, 1024, 3, 1), # 13x13
            nn.Conv2d(1024, num_anchors*(5+num_classes), 1, 1),  # output logits # 13x13
        ])
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        x_stage4, x_stage5 = self.darknet(x)              #(1)
        x = x_stage5
        for i in range(len(self.stage6)):
            x = self.stage6[i](x)                         #(2)
        # print("before passthrough", x_stage4.shape)
        route = self.route_conv(x_stage4) # [B, 64, 26, 26]
        x_stage4 = self.passthrough(route) # [B, 256, 13, 13]         
        # print("after passthrough", x_stage4.shape)

        x = torch.cat([x_stage4, x], dim=1)              
        # print(f'\nx after concatenate\t\t: {x.size()}')
        
        for i in range(len(self.stage7)):               
            x = self.stage7[i](x)
            # print(f'x after stage7 #{i}\t: {x.size()}')    

        x = encode_archor(x, num_anchors=self.num_anchors, num_classes=self.num_classes)
        return x


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, num_classes, device):
    num_batches = len(dataloader)
    dataset_size = len(dataloader.dataset)

    train_correct, train_loss = 0, 0
    object_count = 0
    history = {
        "correct": 0,
        "loss": 0,
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro").to(device),
    }

    model.train()
    for batch , (x, y) in enumerate(dataloader):
        print(f"train_loop-{batch}")
        # if batch == 5:
        #     break

        x = x.to(device)
        y = y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)
        print("\t loss", loss.item())

        # backward
        loss.backward() # compute gradient
        optimizer.step() # update new weight by gradient
        optimizer.zero_grad() # reset grad, since it accumulate grad each batch

        train_loss += loss.item()
        obj_mask = y[..., 4] == 1

        pred_obj = torch.sigmoid(pred[..., 4])
        print("positive objectness:", pred_obj[obj_mask].mean().item())
        print("background objectness:", pred_obj[~obj_mask].mean().item())

        pred_class = pred[..., 5:][obj_mask]   # [N_objects, num_classes]
        true_class = y[..., 5:][obj_mask]       # one-hot, [N_objects, num_classes]
        pred_class_id = pred_class.argmax(dim=-1)
        true_class_id = true_class.argmax(dim=-1)
        history["accuracy"].update(pred_class_id, true_class_id)

        # sum all corrects prediction among anchor boxes and item() convert into float32 
        train_correct += (pred_class_id == true_class_id).sum().item()
        object_count += true_class_id.numel()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"\t loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")

    train_correct = train_correct / object_count if object_count else 0.0
    train_loss  /= num_batches
    history["loss"] = train_loss
    history["correct"] = train_correct
    history["accuracy"] = history["accuracy"].compute().item()
    return history



def test_loop(dataloader, model, loss_fn, num_classes, anchors, device):
    model.eval()
    test_loss = 0
    num_batches = len(dataloader)
    history = {
        "loss": 0,
        "map_metric": MeanAveragePrecision(box_format="xyxy",iou_type="bbox").to(device),
    }

    with torch.no_grad():
        for batch , (X, y) in enumerate(dataloader):
            print(f"test_loop-{batch}")

            # if batch == 5:
            #     break
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # prediction = pred.argmax(1)

            preds = decode_batch_predictions(
                pred,
                anchors,
                conf_threshold=0.001,
                nms_threshold=0.5,
            )
            metric_labels = decode_batch_targets(
                y,
                anchors,
            )
            history["map_metric"].update(preds, metric_labels)

    test_loss /= num_batches
    history["map_metric"] = history["map_metric"].compute()
    history["loss"] = test_loss
    history["map"] = history["map_metric"]["map"].item()
    history["map_50"] = history["map_metric"]["map_50"].item()
    history["map_75"] = history["map_metric"]["map_75"].item()
    history["mar_100"] = history["map_metric"]["mar_100"].item()

    return history

def train_model(epochs, model, train_loader,val_loader, loss_fn, optimizer, scheduler, batch_size, num_classes, anchors, device):
    best_val_map = 0.0

    history = {
        "train_acc": [],
        "train_correct": [],
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
        "val_map_50": [],
        "val_map_75": [],
        "val_mar_100": [],
        "map_metric": [],
    }

    for epoch in range(epochs):
        train_result = train_loop(train_loader, model, loss_fn, optimizer, batch_size, num_classes, device)
        val_result = test_loop(val_loader, model, loss_fn, num_classes, anchors, device) # for evaluate in each epoch
        print(f"Done epoch-{epoch}")
        if val_result['map'] > best_val_map:
            best_val_map = val_result['map']

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_val_map": best_val_map,
            }, "save/stage1_latest.pth")

        # scheduler.step()
        print(
            f"[{epoch+1}/{epochs}] "
            f"train_loss={train_result['loss']:.4f} "
            f"train_correct={train_result['correct']:.4f} "
            f"train_acc={train_result['accuracy']:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"val_mAP={val_result['map']:.4f} "
            f"val_mAP50={val_result['map_50']:.4f} "
            f"val_mAR100={val_result['mar_100']:.4f}"
        )
        history["train_acc"].append(train_result["accuracy"]) # average accuracy among all classes
        history["train_correct"].append(train_result["correct"]) # average by total objects
        history["train_loss"].append(train_result["loss"])
        history["val_loss"].append(val_result["loss"])  # average by total objects
        history["val_map"].append(val_result["map"])
        history["val_map_50"].append(val_result["map_50"])
        history["val_map_75"].append(val_result["map_75"])
        history["val_mar_100"].append(val_result["mar_100"])
        history["map_metric"].append(val_result["map_metric"])

    print("Done epoch training !!!")
    return history

def _decode_yolo_boxes(predictions, anchors, stride, apply_sigmoid_to_centers):
    """Decode YOLO grid predictions or targets into XYXY pixel boxes."""
    _, grid_height, grid_width, num_anchors, _ = predictions.shape
    anchors = torch.as_tensor(
        anchors,
        device=predictions.device,
        dtype=predictions.dtype,
    )

    if anchors.shape != (num_anchors, 2):
        raise ValueError(
            "anchors must have shape "
            f"({num_anchors}, 2), got {tuple(anchors.shape)}"
        )

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_height, device=predictions.device, dtype=predictions.dtype),
        torch.arange(grid_width, device=predictions.device, dtype=predictions.dtype),
        indexing="ij",
    )
    grid_x = grid_x.view(1, grid_height, grid_width, 1)
    grid_y = grid_y.view(1, grid_height, grid_width, 1)

    tx = predictions[..., 0]
    ty = predictions[..., 1]
    if apply_sigmoid_to_centers:
        tx = torch.sigmoid(tx)
        ty = torch.sigmoid(ty)

    center_x = (tx + grid_x) * stride
    center_y = (ty + grid_y) * stride
    width = torch.exp(predictions[..., 2]) * anchors[:, 0].view(1, 1, 1, num_anchors) * stride
    height = torch.exp(predictions[..., 3]) * anchors[:, 1].view(1, 1, 1, num_anchors) * stride

    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def decode_batch_targets(targets, anchors, stride=32):
    """Convert encoded YOLO targets to TorchMetrics ground-truth dictionaries."""
    target_boxes = _decode_yolo_boxes(
        targets,
        anchors,
        stride,
        apply_sigmoid_to_centers=False,
    )
    object_mask = targets[..., 4] == 1
    target_labels = targets[..., 5:].argmax(dim=-1)

    return [
        {
            "boxes": target_boxes[batch_index][object_mask[batch_index]],
            "labels": target_labels[batch_index][object_mask[batch_index]],
        }
        for batch_index in range(targets.shape[0])
    ]


def wrap_yolo_loss(loss_weight=[1, 1, .5, 1], anchors=None, stride=32, ignore_threshold=0.5):
    if anchors is None:
        raise ValueError("anchors are required to compute the YOLO ignore mask")

    def yolo_loss(y_pred, y_true):
        # L box + L class score
        # shape of y label is B, W, H, num_anchors, 5+num_classes
        # shape of y label value is tx,ty, tw,th, objectness, classes ..
        obj_mask = y_true[...,4] == 1 # objectness == 1

        with torch.no_grad():
            pred_boxes = _decode_yolo_boxes(
                y_pred.detach(),
                anchors,
                stride,
                apply_sigmoid_to_centers=True,
            )
            target_boxes = _decode_yolo_boxes(
                y_true,
                anchors,
                stride,
                apply_sigmoid_to_centers=False,
            )
            gt_boxes = [
                target_boxes[batch_index][obj_mask[batch_index]]
                for batch_index in range(y_true.shape[0])
            ]
            ignore_mask = compute_ignore_mask(
                pred_boxes,
                gt_boxes,
                ignore_threshold,
            )

        no_obj_mask = (y_true[..., 4] == 0) & ~ignore_mask

        loss_obj, loss_no_obj, loss_boxes, loss_class_scores = y_pred.new_tensor(0.0), y_pred.new_tensor(0.0), y_pred.new_tensor(0.0), y_pred.new_tensor(0.0)

        pred_tx = torch.sigmoid(y_pred[..., 0])
        pred_ty = torch.sigmoid(y_pred[..., 1])
        target_tx = y_true[..., 0]
        target_ty = y_true[..., 1]

        if obj_mask.any():
            loss_boxes = F.mse_loss(y_pred[...,:4][obj_mask], y_true[...,:4][obj_mask], reduction="mean")

            loss_obj = F.binary_cross_entropy_with_logits(y_pred[..., 4][obj_mask], y_true[..., 4][obj_mask], reduction="mean")
            loss_xy = (
                F.mse_loss(
                    pred_tx[obj_mask],
                    target_tx[obj_mask],
                    reduction="mean",
                )
                +
                F.mse_loss(
                    pred_ty[obj_mask],
                    target_ty[obj_mask],
                    reduction="mean",
                )
            )
            loss_wh = F.mse_loss(y_pred[...,2:4][obj_mask], y_true[...,2:4][obj_mask],reduction="mean")
            loss_boxes = loss_xy + loss_wh

        if no_obj_mask.any():
            loss_no_obj = F.binary_cross_entropy_with_logits(y_pred[..., 4][no_obj_mask], y_true[..., 4][no_obj_mask], reduction="mean")
            noobj_logits = y_pred[..., 4][no_obj_mask]
            noobj_probs = torch.sigmoid(noobj_logits)
            print("noobj count:", noobj_logits.numel())
            print("noobj logits mean:", noobj_logits.mean().item())
            print("noobj logits min :", noobj_logits.min().item())
            print("noobj logits max :", noobj_logits.max().item())
            print("noobj prob mean  :", noobj_probs.mean().item())
            print("noobj prob min   :", noobj_probs.min().item())
            print("noobj prob max   :", noobj_probs.max().item())

            print(
                "loss_noobj:",
                F.binary_cross_entropy_with_logits(
                    noobj_logits,
                    torch.zeros_like(noobj_logits),
                ).item()
            )

        pred_class = y_pred[..., 5:][obj_mask]   # [N_objects, num_classes]
        true_class = y_true[..., 5:][obj_mask]       # one-hot, [N_objects, num_classes]
        loss_class_scores = F.binary_cross_entropy_with_logits(pred_class, true_class, reduction="mean")

        # get max probabilities on each anchor box
        # true_class_id = y_true.argmax(dim=-1)
        # pred_class_id = y_pred.argmax(dim=-1)
        # sum all max probabilities and convert tensor into float32 
        # correct = (pred_class_id == true_class_id).sum().item()

        total_loss = loss_weight[0] * loss_boxes + loss_weight[1] * loss_obj + loss_weight[2] * loss_no_obj + loss_weight[3] * loss_class_scores
        print(
            loss_boxes.item(),
            loss_obj.item(),
            loss_no_obj.item(),
            loss_class_scores.item(),
        )
        return total_loss
    return yolo_loss

def compute_ignore_mask(
    pred_boxes,
    gt_boxes,
    ignore_threshold=0.5,
):
    """
    Compute YOLOv2 ignore mask.

    Args:
        pred_boxes:
            Decoded predicted boxes.
            Shape: [B, S, S, A, 4]
            Format: [x1, y1, x2, y2]

        gt_boxes:
            List of GT boxes, one tensor per image.
            gt_boxes[i].shape = [N_gt_i, 4]
            Format: [x1, y1, x2, y2]

        ignore_threshold:
            Predictions with IoU >= this threshold against ANY GT
            are ignored for no-objectness loss.

    Returns:
        ignore_mask:
            Shape: [B, S, S, A]
            True  = ignore no-objectness loss
            False = normal object/background handling
    """

    B, grid_height, grid_width, A, _ = pred_boxes.shape

    ignore_mask = torch.zeros(
        (B, grid_height, grid_width, A),
        dtype=torch.bool,
        device=pred_boxes.device,
    )

    for b in range(B):

        # ---------------------------------------------------------
        # Predicted boxes for this image
            # [H, W, A, 4] -> [H*W*A, 4]
        # ---------------------------------------------------------
        pred_boxes_b = pred_boxes[b].reshape(-1, 4)
        gt_boxes_b = gt_boxes[b].reshape(-1, 4)

        # ---------------------------------------------------------
        # Ground-truth boxes for this image
        # [N_gt, 4]
        # ---------------------------------------------------------
        gt_boxes_b = gt_boxes_b.to(pred_boxes.device)

        # No GT objects -> nothing to ignore
        if gt_boxes_b.numel() == 0:
            continue

        # ---------------------------------------------------------
        # Pairwise IoU
        #
        # [845, 4] x [N_gt, 4]
        #        ↓
        # [845, N_gt]
        # ---------------------------------------------------------
        iou_matrix = box_iou(
            pred_boxes_b,
            gt_boxes_b,
        )

        # ---------------------------------------------------------
        # For each prediction, find its maximum IoU
        # against ANY GT box.
        #
        # [845, N_gt]
        #       ↓ max(dim=1)
        # [845]
        # ---------------------------------------------------------
        max_iou = iou_matrix.max(dim=1).values
        # ---------------------------------------------------------
        # Ignore predictions that overlap sufficiently with
        # at least one GT object.
        #
        # [845]
        #       ↓
        # [H, W, A]
        # ---------------------------------------------------------
        ignore_mask[b] = (
            max_iou >= ignore_threshold
        ).reshape(grid_height, grid_width, A)

    return ignore_mask
