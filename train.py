#!/usr/bin/env python3

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix per macOS libomp error
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from datasets import load_dataset


# ==========================================================
# 1. Dataset
# ==========================================================

class LicensePlateSegmentationDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, img_size=(256, 256)):
        self.dataset = hf_dataset
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = np.array(sample["image"])
        h, w = image.shape[:2]

        # Crea maschera binaria a partire dalle bounding box
        mask = np.zeros((h, w), dtype=np.uint8)
        for bbox in sample["objects"]["bbox"]:
            x, y, bw, bh = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Applica trasformazioni
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0).float() / 255.0
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            mask = torch.tensor(mask).unsqueeze(0).float() / 255.0

        return image, mask


# ==========================================================
# 2. Data Augmentation
# ==========================================================

def get_transforms():
    train_tfms = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

    val_tfms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])
    return train_tfms, val_tfms


# ==========================================================
# 3. Modello
# ==========================================================

def get_model():
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model


# ==========================================================
# 4. Training Loop
# ==========================================================
dice_loss_fn = smp.losses.DiceLoss(mode="binary")
bce_loss_fn = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return dice_loss_fn(pred, target) + bce_loss_fn(pred, target)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, device="cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    loss_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, combined_loss, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train: {avg_train:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unetpp.pth")
            print("✅ Model saved (best)")

        # Append to loss list for plotting
        loss_list.append((avg_train, val_loss))
        
        # Create loss plot
        plt.figure(figsize=(10, 5))
        plt.plot([l[0] for l in loss_list], label="Train Loss")
        plt.plot([l[1] for l in loss_list], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig("loss_curve.png")
        plt.close()

    print("Training complete!")


# ==========================================================
# 5. Validation
# ==========================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


# ==========================================================
# 6. Visualizzazione esempio
# ==========================================================

def visualize_result(model, dataset, device="cuda"):
    model.eval()
    img, mask = dataset[0]
    with torch.no_grad():
        pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title("Immagine")

    plt.subplot(1, 3, 2)
    plt.imshow(mask[0], cmap="gray")
    plt.title("Maschera GT")

    plt.subplot(1, 3, 3)
    plt.imshow(pred > 0.5, cmap="gray")
    plt.title("Predizione")
    plt.show()


# ==========================================================
# 7. Main
# ==========================================================

def main():
    print("🚀 Caricamento dataset...")
    ds = load_dataset("keremberke/license-plate-object-detection", name="full")
    hf_dataset = ds
    train_hf = ds["train"]
    test_hf = ds["test"]
    val_hf = ds["validation"]

    train_tfms, val_tfms = get_transforms()
    full_dataset = LicensePlateSegmentationDataset(hf_dataset, transform=None)

    # Split 80/20
    #train_size = int(0.8 * len(full_dataset))
    #val_size = len(full_dataset) - train_size
    #train_hf, val_hf = random_split(hf_dataset, [train_size, val_size])
    train_size = len(train_hf)
    val_size = len(val_hf)
    test_size = len(test_hf)
    print(f"Dataset size: Train={train_size}, Val={val_size}, Test={test_size}")

    train_dataset = LicensePlateSegmentationDataset(train_hf, transform=train_tfms)
    val_dataset = LicensePlateSegmentationDataset(val_hf, transform=val_tfms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🧠 Training on {device}")

    model = get_model()
    train_model(model, train_loader, val_loader, num_epochs=15, device=device)

    # Visualizza esempio
    model.load_state_dict(torch.load("best_unetpp.pth", map_location=device))
    visualize_result(model, val_dataset, device=device)


if __name__ == "__main__":
    main()