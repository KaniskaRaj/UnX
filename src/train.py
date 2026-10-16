import os
import argparse
import random
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torch.amp import autocast
import pandas as pd
import numpy as np
import csv

from dataset import FastCXRDataset, get_cxr_transforms, DISEASES

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="UnCXR: Train DenseNet-121")
    parser.add_argument("--seed", type=int, default=10, help="Random seed (10 for Sinchana, 20 for Shashi, 30 for Ananya)")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_dir", type=str, default="/content/images_224", help="Local directory with 224x224 images")
    parser.add_argument("--data_dir", type=str, default="/content/UnCXR/data", help="Directory with train.csv and val.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/UnCXR_Dataset/checkpoints", help="Path to save checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training DenseNet-121 (Seed {args.seed}) on {device} ===")

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))

    transform = get_cxr_transforms()
    train_loader = DataLoader(
        FastCXRDataset(train_df, args.img_dir, transform),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        FastCXRDataset(val_df, args.img_dir, transform),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    label_counts = train_df[DISEASES].sum(axis=0).to_numpy(dtype=np.float64)
    pos_weight = (len(train_df) - label_counts) / np.maximum(label_counts, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))

    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier.in_features, len(DISEASES))
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"seed{args.seed}.pt")
    history_path = os.path.join(args.checkpoint_dir, f"seed{args.seed}_training_history.csv")

    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(args.epochs):
        model.train()
        t_loss, count = 0.0, 0
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.detach().item() * images.size(0)
            count += images.size(0)

        train_loss = t_loss / max(count, 1)

        model.eval()
        v_loss, v_count = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()
                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, labels)
                v_loss += loss.item() * images.size(0)
                v_count += images.size(0)
        val_loss = v_loss / max(v_count, 1)

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "seed": args.seed,
                "diseases": DISEASES
            }, best_ckpt_path)
            print(f"  * Saved new best model checkpoint -> {best_ckpt_path}")

    print(f"Training for Seed {args.seed} finished successfully!")

if __name__ == "__main__":
    main()
