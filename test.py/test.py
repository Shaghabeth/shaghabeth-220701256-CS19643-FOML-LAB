import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from glob import glob
import torchvision.transforms as transforms

from dataset import segDataset, to_device, get_default_device
from loss import FocalLoss
from Unet import UNet
from utils import acc

# ------------------------------------
# Transformations
# ------------------------------------
color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
t = transforms.Compose([color_shift, blurriness])

# ------------------------------------
# Load Dataset
# ------------------------------------
dataset_path = os.path.join(os.getcwd(), "Semantic segmentation dataset")
dataset = segDataset(dataset_path, training=False, transform=t)

test_num = int(0.1 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [len(dataset) - test_num, test_num], generator=torch.Generator().manual_seed(101))

BATCH_SIZE = 4
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ------------------------------------
# Device Setup
# ------------------------------------
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
test_dataloader = DeviceDataLoader(test_dataloader, device)

# ------------------------------------
# Load Best or Latest Model
# ------------------------------------
model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)

# Try to load the latest model
model_files = sorted(glob(os.path.join("saved_models", "unet_epoch_*.pt")))
model_path = None

if model_files:
    model_path = model_files[-1]
elif os.path.exists(os.path.join("saved_models", "unet_best.pt")):
    model_path = os.path.join("saved_models", "unet_best.pt")

if model_path:
    print(f"[INFO] Loading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    raise FileNotFoundError("❌ No model found in saved_models/. Please run train.py first.")

model.eval()

# ------------------------------------
# Inference & Visualization
# ------------------------------------
for batch_i, (x, y) in enumerate(test_dataloader):
    for j in range(len(x)):
        with torch.no_grad():
            result = model(x[j:j+1])
        pred_mask = torch.argmax(result, axis=1).cpu().detach().numpy()[0]
        input_image = np.moveaxis(x[j].cpu().numpy(), 0, -1) * 255
        input_image = input_image.astype(np.uint8)
        gt_mask = y[j].cpu().numpy()

        # Plotting
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(input_image)

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(gt_mask)

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask)

        plt.tight_layout()
        plt.show()
