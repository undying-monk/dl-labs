import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from torchmetrics.classification import (
    MulticlassAccuracy,
)
from torchmetrics.detection import MeanAveragePrecision

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

    
class EncoderBlock(nn.Module):
    """Standard Convolution -> Batch Normalization -> Leaky ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # double conv
        # Input Pixels:  [1]  [2]  [3]  [4]  [5]   <- 5 total pixels!
        #          \   |   /   \  |  /
        # Layer 1 Nodes:   [Node A] [Node X] [Node B]  <- Each node saw 3 pixels
        #                     \        |        /
        # Layer 2 Node:         [  Super Node  ]       <- Looks at 3 nodes, but indirectly 
        #                                                 sees ALL 5 input pixels!
        self.features = nn.Sequential( 
            DoubleConvBlock(in_channels, out_channels, kernel_size, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class DecoderBlock(nn.Module):
    """Standard Convolution -> Batch Normalization -> Leaky ReLU block"""
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=3):
        super().__init__()
        # Double spatial resolution and reduce channels using Tranpose Convolution
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) 

        # Concatenation adds the decoder and skip channels
        self.block = nn.Sequential(
            DoubleConvBlock(skip_channels+out_channels, out_channels, kernel_size, stride=1, padding=1),
        )
    def forward(self, x, x_encoder):
        x = self.up(x)

        # Concatenate along channel dimension
        x = torch.cat([x, x_encoder], dim=1)
        return self.block(x)

class EncoderBlock(nn.Module):
    """Standard Convolution -> Batch Normalization -> Leaky ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # double conv
        # Input Pixels:  [1]  [2]  [3]  [4]  [5]   <- 5 total pixels!
        #          \   |   /   \  |  /
        # Layer 1 Nodes:   [Node A] [Node X] [Node B]  <- Each node saw 3 pixels
        #                     \        |        /
        # Layer 2 Node:         [  Super Node  ]       <- Looks at 3 nodes, but indirectly 
        #                                                 sees ALL 5 input pixels!
        self.features = nn.Sequential( 
            DoubleConvBlock(in_channels, out_channels, kernel_size, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        return x

# [B,C,H,W] => [B]
def encode_to_categories(pred):
    return pred[..., 1]
    
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder_stage_1 = EncoderBlock(3,64) # 256x256x64
        self.pool_1 = nn.MaxPool2d(2)    # 128x128x64
        self.encoder_stage_2 = EncoderBlock(64, 128) # 128x128x128
        self.pool_2 = nn.MaxPool2d(2)    # 64x64x128
        self.encoder_stage_3 = EncoderBlock(128,256) # 64x64x256
        self.pool_3 = nn.MaxPool2d(2)    # 32X32X256
        self.encoder_stage_4 = EncoderBlock(256,512) # 32X32X512
        self.pool_4 = nn.MaxPool2d(2)    # 16x16x512

        self.bottle_neck = EncoderBlock(512,1024) # 16x16x1024 => bottle neck
        self.decoder_stage_1 = DecoderBlock(1024,512,512) # 32x32x512, 1024 -> 512 + 512 -> 512
        self.decoder_stage_2 = DecoderBlock(512,256,256) # 64x64x256
        self.decoder_stage_3 = DecoderBlock(256,128,128) # 128x128x128
        self.decoder_stage_4 = DecoderBlock(128,64,64) # 256x256x64
        self.final = nn.Conv2d(64, num_classes, 1) # 256x256x N_classes

    def forward(self, x):
        # print("before encoder_stage_1", x.shape, type(x))
        x = self.encoder_stage_1(x)
        # print("after encoder_stage_1", x.shape)
        x_1 = x.clone() 
        x = self.pool_1(x)

        x = self.encoder_stage_2(x)
        x_2 = x.clone() 
        x = self.pool_2(x)

        x = self.encoder_stage_3(x)
        x_3 = x.clone() 
        x = self.pool_3(x)

        x = self.encoder_stage_4(x)
        x_4 = x.clone() 
        x = self.pool_4(x)

        x = self.bottle_neck(x)

        # print("before decoder_stage_1", x.shape, type(x))
        x = self.decoder_stage_1(x, x_4)
        # print("before decoder_stage_2", x.shape, type(x))
        x = self.decoder_stage_2(x, x_3)
        x = self.decoder_stage_3(x, x_2)
        x = self.decoder_stage_4(x, x_1)
        x = self.final(x)
        # x = encode_to_categories(x)
        return x # [H,W,classes], [B, C, H, W] => for epoch, so use argmax to retrieve [B,H,W]


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, num_classes, device):
    num_batches = len(dataloader)
    dataset_size = len(dataloader.dataset)

    train_correct, train_loss = 0, 0
    object_count = 0
    history = {
        "correct": 0,
        "loss": 0,
        "accuracy": 0
        # "accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro").to(device),
    }

    model.train()
    for batch , (x, y) in enumerate(dataloader):
        print(f"train_loop-{batch}")
        if batch == 5:
            break

        x = x.to(device) # [3,256,256]
        y = y.to(device) # [B, 1,256,256]
        y = y.squeeze(dim=1) # [B,256,256]
        
        # forward
        pred = model(x) # [B, num_classes, 256,256]
        loss = loss_fn(pred, y)
        print("\t loss", loss.item())

        # backward
        loss.backward() # compute gradient
        optimizer.step() # update new weight by gradient
        optimizer.zero_grad() # reset grad, since it accumulate grad each batch

        train_loss += loss.item()

        pred_class_id = torch.argmax(pred, dim=1) # [B, num_classes, 256,256] => [B, 256,256], row_wise argmax
        train_correct += (pred_class_id == y).sum().item()
        object_count += y.numel()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"\t loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")

    train_correct = train_correct / object_count if object_count else 0.0
    train_loss  /= num_batches
    history["loss"] = train_loss
    history["correct"] = train_correct
    # history["accuracy"] = history["accuracy"].compute().item()
    return history



def test_loop(dataloader, model, loss_fn, num_classes, device):
    model.eval()
    test_correct, test_loss = 0, 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    history = {
        "loss": 0,
        "correct": 0,
    }

    with torch.no_grad():
        for batch , (X, y) in enumerate(dataloader):
            print(f"test_loop-{batch}")

            if batch == 5:
                break
            X = X.to(device) # [3,256,256]
            y = y.to(device)  # [B, 1,256,256]
            y = y.squeeze(dim=1) # [B,256,256]

            pred = model(X) # [B, num_classes, 256,256]
            test_loss += loss_fn(pred, y).item()
            print("\t test_loss", test_loss)

            pred_class_id = torch.argmax(pred, dim=1)
            test_correct += (pred_class_id == y).sum().item() # sum predictions of each batch

    test_loss /= num_batches
    test_correct /= size
    history["loss"] = test_loss
    history["correct"] = test_correct

    return history

def save_checkpoint(path, epoch, model, optimizer=None, scheduler=None, metrics=None):
    """Save a resumable YOLOv2 training checkpoint."""
    model_config = {}
    if hasattr(model, "num_classes"):
        model_config["num_classes"] = model.num_classes

    checkpoint = {
        "format_version": 1,
        "epoch": epoch,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    """Load a checkpoint produced by :func:`save_checkpoint`."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    required_keys = {"format_version", "epoch", "model_state_dict", "metrics"}
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise ValueError(
            f"Invalid YOLOv2 checkpoint at {path}: missing keys {sorted(missing_keys)}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

def train_model(
    epochs,
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    batch_size,
    num_classes,
    device,
    checkpoint_path="save/latest.pth",
    best_checkpoint_path="save/best.pth",
    resume=True,
):
    start_epoch = 0

    history = {
        "train_acc": [],
        "train_correct": [],
        "train_loss": [],
        "val_loss": [],
        "val_correct": [],
    }


    best_val_correct = 0
    if resume and Path(checkpoint_path).is_file():
        checkpoint = load_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = checkpoint["epoch"] + 1
        checkpoint_metrics = checkpoint["metrics"]
        best_val_correct = checkpoint_metrics.get("best_val_correct", 0.0)
        history = checkpoint_metrics.get("history", history)
        print(f"Resuming training from epoch {start_epoch}.")

    for epoch in range(start_epoch, epochs):
        train_result = train_loop(train_loader, model, loss_fn, optimizer, batch_size, num_classes, device)
        val_result = test_loop(val_loader, model, loss_fn, num_classes, device) # for evaluate in each epoch
        print(f"Done epoch-{epoch}")
        is_best = val_result["correct"] > best_val_correct
        if is_best:
            best_val_correct = val_result["correct"]

        print(
            f"[{epoch+1}/{epochs}] "
            f"train_loss={train_result['loss']:.4f} "
            f"train_correct={train_result['correct']:.4f} "
            f"train_acc={train_result['accuracy']:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"val_correct={val_result['correct']:.4f} "
        )
        history["train_acc"].append(train_result["accuracy"]) # average accuracy among all classes
        history["train_correct"].append(train_result["correct"]) # average by total objects
        history["train_loss"].append(train_result["loss"])
        history["val_loss"].append(val_result["loss"])  # average by total objects
        history["val_correct"].append(val_result["correct"])  # average by total objects

        if scheduler is not None:
            scheduler.step()

        checkpoint_metrics = {
            "best_val_correct": best_val_correct,
            "history": history,
        }
        save_checkpoint(
            checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            metrics=checkpoint_metrics,
        )
        if is_best:
            save_checkpoint(
                best_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                metrics=checkpoint_metrics,
            )

    print("Done epoch training !!!")
    return history