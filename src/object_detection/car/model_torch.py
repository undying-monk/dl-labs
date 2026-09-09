import numpy as np
import torch
from torch import nn
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from torch.utils.data import Dataset, DataLoader
import math
from util import find_highest_iou_anchor, encode_archor

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
            ConvBlock(1024, num_anchors*(5+num_classes), 1, 1)    # output layer # 13x13
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


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    train_correct, train_loss = 0, 0

    model.train()
    for batch , (x, y) in enumerate(dataloader):
        print(f"train_loop-{batch}", type(x), type(y))
        x = x.to(device)
        y = y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)
        print("loss", loss)

        # backward
        loss.backward() # compute gradient
        optimizer.step() # update new weight by gradient
        optimizer.zero_grad() # reset grad, since it accumulate grad each batch

        train_loss += loss.item()

        # Get the index of the highest probability, dim=-1 apply for last dimension which is num_classes
        prediction = pred.argmax(dim=-1)
        print(prediction.shape, y.shape)

        obj_mask = y[..., 4] == 1
        pred_class = pred[..., 5:][obj_mask]   # [N_objects, num_classes]
        true_class = y[..., 5:][obj_mask]       # one-hot, [N_objects, num_classes]
        pred_class_id = pred_class.argmax(dim=-1)
        true_class_id = true_class.argmax(dim=-1)

        # sum all corrects prediction among anchor boxes and item() convert into float32 
        train_correct += (pred_class_id == true_class_id).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_correct /= size
    train_loss  /= num_batches
    return train_correct, train_loss

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    test_correct, test_loss = 0, 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    y_pred, labels = [], []

    with torch.no_grad():
        for X, y in dataloader: # run each batch
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            prediction = pred.argmax(1)
            test_correct += (prediction == y).sum().item() # sum predictions of each batch
            y_pred.append(prediction)
            labels.append(y)

    test_loss /= num_batches
    test_correct /= size
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_correct:.4f}")

    return test_correct, test_loss, (torch.cat(y_pred, dim=0), torch.cat(labels, dim=0))


def train_model(epochs, model, train_loader, val_loader, loss_fn, optimizer, scheduler, batch_size, device, num_classes):
    # accuracy = MulticlassAccuracy(num_classes=num_classes)
    history = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
        "val_precision": MulticlassPrecision(
            num_classes=num_classes,
            average="macro"
        ).to(device),
        "val_recall":  MulticlassRecall(
            num_classes=num_classes,
            average="macro"
        ).to(device),
        "val_f1": MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        ).to(device),
    }
    best_val_correct = 0.0

    for epoch in range(epochs):

        train_correct,train_loss = train_loop(train_loader, model, loss_fn, optimizer, batch_size, device)
        print("train_correct", train_correct, train_loss)
        val_correct,val_loss, (y_pred, y) = test_loop(val_loader, model, loss_fn, device) # for evaluate in each epoch
        history["train_acc"].append(train_correct)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_correct)
        history["val_loss"].append(val_loss)

        # -----------------------
        # Save best model
        # -----------------------
        if val_correct > best_val_correct:
            best_val_correct = val_correct

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
                    if scheduler is not None else None,
                "val_loss": val_loss,
                "val_acc": val_correct,
            }, "save/stage1_latest.pth")

        # accuracy.update(y_pred, y)
        history["val_precision"].update(y_pred, y)
        history["val_recall"].update(y_pred, y)
        history["val_f1"].update(y_pred, y)

        # scheduler.step()
        print("Done!")
        

    history["val_precision"] = history["val_precision"].compute().item()
    history["val_recall"] = history["val_recall"].compute().item()
    history["val_f1"] = history["val_f1"].compute().item()

    return history

def wrap_yolo_loss(loss_weight=[1, 1, 1, 1]):
    def yolo_loss(y_true,y_pred):
        # L box + L class score
        # shape of y label is B, W, H, num_anchors, 5+num_classes
        # shape of y label value is tx,ty, tw,th, objectness, classes ..
        filter = y_true[...,4] == 1 # objectness == 1
        no_obj_filter = y_true[...,4] == 0 # objectness == 1

        loss_boxes = nn.MSELoss()(y_pred[...,:4][filter], y_true[...,:4][filter])
        loss_obj = nn.BCEWithLogitsLoss()(y_pred[..., 4][filter], y_true[..., 4][filter])
        loss_no_obj = nn.BCEWithLogitsLoss()(y_pred[..., 4][no_obj_filter], y_true[..., 4][no_obj_filter])

        pred_class = y_pred[..., 5:][filter]   # [N_objects, num_classes]
        true_class = y_true[..., 5:][filter]       # one-hot, [N_objects, num_classes]
        loss_class_scores = nn.BCEWithLogitsLoss(reduction="sum")(pred_class, true_class)

        # get max probabilities on each anchor box
        true_class_id = y_true.argmax(dim=-1)
        pred_class_id = y_pred.argmax(dim=-1)

        # sum all max probabilities and convert tensor into float32 
        correct = (pred_class_id == true_class_id).sum().item()
        total_loss = loss_weight[0] * loss_boxes + loss_weight[1] * loss_obj + loss_weight[2] * loss_no_obj + loss_weight[3] * loss_class_scores

        return total_loss
    return yolo_loss


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
        
        # 3. Explicitly set values at the specified indices
        # If indices is a tuple of coordinates (e.g., (row_array, col_array)), 
        # PyTorch handles advanced multi-dimensional index assignment naturally.
        for i, box in enumerate(bounding_boxes):
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
    
            # find central point to find grid cell
            midpoint_x = (xmax - xmin) /2
            midpoint_y = (ymax - ymin) /2
            b_w = (xmax - xmin) / self.grid_width # calculate width and convert into grid units
            b_h = (ymax - ymin) / self.grid_height # calculate width and convert into grid units
    
            grid_x = int(midpoint_x / self.grid_width)
            grid_y = int(midpoint_y / self.grid_height)
    
            t_x = (midpoint_x % self.grid_width) / self.grid_width # position inside cell
            t_y = (midpoint_y % self.grid_height) / self.grid_height
    
            # print("position",t_x,t_y, midpoint_x, midpoint_y, grid_height)
    
            # find match anchor box with label bounding box
            best_anchor = find_highest_iou_anchor([b_w,b_h], self.anchors)
            # print("bounding box", f"[{b_w:.2f}, {b_h:.2f}]", "- anchor", anchors[best_anchor])
            t_w = math.log(b_w/self.anchors[best_anchor][0] )
            t_h = math.log(b_h/self.anchors[best_anchor][1] )
    
            # position
            target_tensor[grid_x, grid_y, best_anchor, 0] = t_x 
            target_tensor[grid_x, grid_y, best_anchor, 1] = t_y
            # size
            target_tensor[grid_x, grid_y, best_anchor, 2] = t_w  # width
            target_tensor[grid_x, grid_y, best_anchor, 3] = t_h  # height
    
            # object exists
            target_tensor[grid_x, grid_y, best_anchor, 4] = 1.0
    
            # class
            target_tensor[grid_x, grid_y, best_anchor, 5 + labels[i]] = 1.0
            # print("label_data", label_data[grid_x, grid_y, best_anchor, :]) 
        return target_tensor