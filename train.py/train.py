import os
import sys
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from dataset import segDataset, to_device, get_default_device
from loss import FocalLoss
from Unet import UNet
from utils import acc

# ----------------------------------------
# Data Augmentation
# ----------------------------------------
color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
t = transforms.Compose([color_shift, blurriness])

# ----------------------------------------
# Dataset Loading
# ----------------------------------------
dataset_path = os.path.join(os.getcwd(), "Semantic segmentation dataset")
dataset = segDataset(dataset_path, training=True, transform=t)

test_num = int(0.1 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [len(dataset) - test_num, test_num], generator=torch.Generator().manual_seed(101))

BATCH_SIZE = 4  # You can reduce to 2 or 1 if system is slow
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------------------------
# Device + DataLoader Wrappers
# ----------------------------------------
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

device = get_default_device()
train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

# ----------------------------------------
# Model, Loss, Optimizer, Scheduler
# ----------------------------------------
model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
criterion = FocalLoss(gamma=3/4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# ----------------------------------------
# Setup for saving models
# ----------------------------------------
os.makedirs("saved_models", exist_ok=True)
min_val_loss = float("inf")

# ----------------------------------------
# Training Loop
# ----------------------------------------
N_EPOCHS = 50
plot_losses = []

for epoch in range(N_EPOCHS):
    model.train()
    train_loss, train_acc = [], []

    for batch_i, (x, y) in enumerate(train_loader):
        pred = model(x)
        loss = criterion(pred, y)

        if torch.isnan(loss):
            print(f"⚠️ Skipping batch {batch_i+1} due to NaN in loss.")
            continue

        print(f"\n→ Doing backward for loss: {loss.item():.4f}")  # Debug line
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_acc.append(acc(y, pred).item())

        sys.stdout.write(f"\r[Epoch {epoch+1}/{N_EPOCHS}] [Batch {batch_i+1}/{len(train_loader)}] [Loss: {loss.item():.4f}]")
        sys.stdout.flush()  # Ensure immediate output

    # Validation
    model.eval()
    val_loss, val_acc = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            loss = criterion(pred, y)
            val_loss.append(loss.item())
            val_acc.append(acc(y, pred).item())

    avg_train_loss = np.mean(train_loss)
    avg_val_loss = np.mean(val_loss)
    avg_train_acc = np.mean(train_acc)
    avg_val_acc = np.mean(val_acc)

    print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Train Acc = {avg_train_acc:.2f}, Val Acc = {avg_val_acc:.2f}")

    # Save model at every epoch
    torch.save(model.state_dict(), f"saved_models/unet_epoch_{epoch+1}_{avg_val_loss:.5f}.pt")

    # Save best model
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        torch.save(model.state_dict(), "saved_models/unet_best.pt")
        print("[✓] Best model updated!")

    plot_losses.append([epoch+1, avg_train_loss, avg_val_loss])
    lr_scheduler.step()

# ----------------------------------------
# Plot loss graph (optional)
# ----------------------------------------
plot_losses = np.array(plot_losses)
plt.plot(plot_losses[:, 0], plot_losses[:, 1], label="Train Loss")
plt.plot(plot_losses[:, 0], plot_losses[:, 2], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.show()
