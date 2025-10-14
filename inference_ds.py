#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from datasets import load_dataset
import random
import cv2

# ==========================================================
# 1. Trasformazioni
# ==========================================================
def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

# ==========================================================
# 2. Modello
# ==========================================================
def get_model(path, device="cuda"):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==========================================================
# 3. Inferenza
# ==========================================================
def predict(model, image, transform, device="cuda"):
    augmented = transform(image=image)
    img_tensor = augmented["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0,0].cpu().numpy()
    return (pred > 0.5).astype(np.uint8)

# ==========================================================
# 4. Visualizzazione
# ==========================================================
def visualize(image, mask, index, save_dir="output", file_prefix="license_plate"):
    os.makedirs(save_dir, exist_ok=True)

    # Ridimensiona la maschera alla dimensione originale
    h, w = image.shape[:2]
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # Overlay: sfondo nero
    overlay = image.copy()
    overlay[mask_resized == 0] = 0

    # Visualizzazione
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title("Immagine originale")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(overlay)
    plt.title("Targa evidenziata")
    plt.axis("off")

    # 3. Estrai solo la targa
    ys, xs = np.where(mask_resized > 0)
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        license_plate = image[y_min:y_max+1, x_min:x_max+1]

        plt.subplot(1,3,3)
        plt.imshow(license_plate)
        plt.title("Targa ritagliata")
        plt.axis("off")

        # Salva su file
        save_path = os.path.join(save_dir, f"{file_prefix}_{index}.png")
        cv2.imwrite(save_path, cv2.cvtColor(license_plate, cv2.COLOR_RGB2BGR))
        print(f"Targa salvata: {save_path}")
    else:
        plt.subplot(1,3,3)
        plt.text(0.5,0.5,"Maschera vuota", ha='center', va='center')
        plt.axis("off")

    plt.show()

# ==========================================================
# 5. Main
# ==========================================================
def main():
    import time
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Inferenza su device: {device}")

    # Carica modello
    model = get_model("unetpp_11epochs.pth", device=device)
    transform = get_transform()

    # Carica dataset test
    ds = load_dataset("keremberke/license-plate-object-detection", name="full")
    dataset = ds["test"]

    # Seleziona 20 immagini casuali
    indices = random.sample(range(len(dataset)), 20)

    total_time = 0.0

    for i, idx in enumerate(indices, 1):
        sample = dataset[idx]
        img = np.array(sample["image"])

        start_time = time.time()
        mask_pred = predict(model, img, transform, device=device)
        elapsed = time.time() - start_time
        total_time += elapsed

        print(f" - [{i}/{len(indices)}] Tempo: {elapsed:.3f} secondi")

        visualize(img, mask_pred, idx)

    avg_time = total_time / len(indices)
    print(f"\n Inferenza completata su {len(indices)} immagini.")
    print(f" - Tempo medio per immagine: {avg_time:.3f} s")

if __name__ == "__main__":
    main()