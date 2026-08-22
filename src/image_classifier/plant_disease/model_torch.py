from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), # 26
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 13
            nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, 3), # 11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 5
            nn.Dropout2d(0.2),

            nn.Conv2d(64,128, 3), # 11
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 5
            nn.Dropout2d(0.25),

            nn.Conv2d(128,256, 3), # 11
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 5
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 1 * 1, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x) 
        x = torch.flatten(x, 1)    # Output shape: [Batch, 64] (Flattens 1x1 spatial dims)
        x = self.classifier(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    train_correct, train_loss = 0, 0

    model.train()
    for batch , (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward
        loss.backward() # compute gradient
        optimizer.step() # update new weight by gradient
        optimizer.zero_grad() # reset grad, since it accumulate grad each batch

        train_loss += loss.item()
        prediction = pred.argmax(dim=1)
        train_correct += (prediction == y).sum().item()
 
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


def train_model(epochs, model, train_loader, val_loader, loss_fn, optimizer, batch_size, device, num_classes):
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

    for t in range(epochs):
        train_correct,train_loss = train_loop(train_loader, model, loss_fn, optimizer, batch_size, device)
        test_correct,test_loss, (y_pred, y) = test_loop(val_loader, model, loss_fn, device) # for evaluate in each epoch
        history["train_acc"].append(train_correct)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(test_correct)
        history["val_loss"].append(test_loss)

        # accuracy.update(y_pred, y)
        history["val_precision"].update(y_pred, y)
        history["val_recall"].update(y_pred, y)
        history["val_f1"].update(y_pred, y)

        print("Done!")

    history["val_precision"] = history["val_precision"].compute().item()
    history["val_recall"] = history["val_recall"].compute().item()
    history["val_f1"] = history["val_f1"].compute().item()

    return history