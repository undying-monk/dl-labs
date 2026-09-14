import numpy as np
import torch
from torch import nn
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import math
from util import decode_predictions, encode_archor, decode_batch_predictions

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
        print(f"train_loop-{batch}")
        # if batch == 1:
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

        # Get the index of the highest probability, dim=-1 apply for last dimension which is num_classes
        # prediction = pred.argmax(dim=-1)
        # print(prediction.shape, y.shape)

        obj_mask = y[..., 4] == 1
        pred_class = pred[..., 5:][obj_mask]   # [N_objects, num_classes]
        true_class = y[..., 5:][obj_mask]       # one-hot, [N_objects, num_classes]
        pred_class_id = pred_class.argmax(dim=-1)
        true_class_id = true_class.argmax(dim=-1)

        # sum all corrects prediction among anchor boxes and item() convert into float32 
        train_correct += (pred_class_id == true_class_id).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"\t loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
            # prediction = pred.argmax(1)

            obj_mask = y[..., 4] == 1
            pred_class = pred[..., 5:][obj_mask]   # [N_objects, num_classes]
            true_class = y[..., 5:][obj_mask]       # one-hot, [N_objects, num_classes]
            pred_class_id = pred_class.argmax(dim=-1)
            true_class_id = true_class.argmax(dim=-1)

            # sum all corrects prediction among anchor boxes and item() convert into float32 
            test_correct += (pred_class_id == true_class_id).sum().item() # sum predictions of each batch
            
            y_pred.append(pred)
            labels.append(y)

    test_loss /= num_batches
    test_correct /= size
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_correct:.4f}")

    return test_correct, test_loss, (torch.cat(y_pred, dim=0), torch.cat(labels, dim=0))

def validate_model(epochs, model, val_loader, loss_fn, optimizer, scheduler,device, anchors, metric):
    history = {
        "val_acc": [],
        "val_loss": [],
        "metrics": metric
    }
    best_val_correct = 0.0
    metric.reset()
    preds = []
    metric_labels = []
    for epoch in range(epochs):
        val_correct,val_loss, (y_pred, y) = test_loop(val_loader, model, loss_fn, device) # for evaluate in each epoch
        print(y_pred.shape, y.shape)
        history["val_acc"].append(val_correct)
        history["val_loss"].append(val_loss)

        preds = decode_batch_predictions(
            y_pred,
            anchors,
            conf_threshold=0.001,
            nms_threshold=0.5,
        )
        metric_labels = decode_batch_predictions(
            y_pred,
            anchors,
            conf_threshold=0.001,
            nms_threshold=0.5,
        )
        print("decode_batch_predictions", preds)


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

        # scheduler.step()
        print("Done! Validation")


    metric.update(preds, metric_labels)
    history["metrics"] = metric.compute()   
    return history

def train_model(epochs, model, train_loader, loss_fn, optimizer, scheduler, batch_size, device):
    history = {
        "train_acc": [],
        "train_loss": [],
    }
    best_train_correct = 0.0
    preds = []
    metric_labels = []
    for epoch in range(epochs):

        train_correct,train_loss = train_loop(train_loader, model, loss_fn, optimizer, batch_size, device)
        print("train_correct", train_correct, train_loss)
        history["train_acc"].append(train_correct)
        history["train_loss"].append(train_loss)


        # for batch_index in range(y_pred.shape[0]):
        #     print("batch_index", batch_index)

        #     pred_boxes, pred_scores, pred_labels  = decode_predictions(y_pred[batch_index], anchors)
        #     boxes, _, labels  = decode_predictions(y[batch_index], anchors)
        #     metric_labels.append({
        #         "boxes": boxes.float(),
        #         "labels": labels.long(),
        #     })
        #     preds.append( {
        #             "boxes": pred_boxes,
        #             "scores": pred_scores,
        #             "labels": pred_labels,
        #     })

        # -----------------------
        # Save best model
        # -----------------------
        if train_correct > best_train_correct:
            best_train_correct = train_correct

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
                    if scheduler is not None else None,
            }, "save/stage1_latest.pth")

        # scheduler.step()
        print("Done! Train")
    return history

def wrap_yolo_loss(loss_weight=[1, 1, 1, 1]):
    def yolo_loss(y_pred, y_true):
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