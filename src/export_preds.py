import os
import argparse
import torch
import numpy as np
import pandas as pd
from torchvision import models
from torch.utils.data import DataLoader

from dataset import FastCXRDataset, get_cxr_transforms, DISEASES

def export_split(model, split_name, args, transform, device):
    csv_path = os.path.join(args.data_dir, f"{split_name}.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping.")
        return

    loader = DataLoader(
        FastCXRDataset(csv_path, args.img_dir, transform),
        batch_size=64, shuffle=False, num_workers=2
    )

    all_logits = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    out_file = os.path.join(args.preds_dir, f"{split_name}_seed{args.seed}_logits.npy")
    np.save(out_file, logits_arr)
    print(f"Exported {split_name} logits (Shape: {logits_arr.shape}) -> {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Export raw logits as .npy arrays")
    parser.add_argument("--seed", type=int, default=10, help="Model seed to evaluate")
    parser.add_argument("--img_dir", type=str, default="/content/images_224", help="Local images directory")
    parser.add_argument("--data_dir", type=str, default="/content/UnCXR/data", help="Directory containing split CSVs")
    parser.add_argument("--preds_dir", type=str, default="/content/drive/MyDrive/UnCXR_Dataset/preds", help="Output directory for predictions")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/UnCXR_Dataset/checkpoints", help="Directory with checkpoints")
    args = parser.parse_args()

    os.makedirs(args.preds_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.checkpoint_dir, f"seed{args.seed}.pt")

    transform = get_cxr_transforms()
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(model.classifier.in_features, len(DISEASES))
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded weights: {ckpt_path}")
    export_split(model, "val", args, transform, device)
    export_split(model, "test", args, transform, device)

if __name__ == "__main__":
    main()
