import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Pneumonia", "Pneumothorax"
]

class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["Image Index"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def enable_dropout_only(model):
    """Keep all layers in eval mode except Dropout layers."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

def build_model(num_classes=7, dropout_rate=0.2):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model

def run_mc_dropout(model, dataloader, device, num_samples=25):
    model.eval()
    enable_dropout_only(model)

    all_sample_probs = []

    with torch.no_grad():
        for sample_idx in range(num_samples):
            probs_list = []
            for images in tqdm(dataloader, desc=f"MC Sample {sample_idx+1}/{num_samples}"):
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())
            all_sample_probs.append(np.concatenate(probs_list, axis=0))

    stacked = np.stack(all_sample_probs, axis=0)
    mean_probs = np.mean(stacked, axis=0)
    variance_uncertainty = np.var(stacked, axis=0)

    return mean_probs, variance_uncertainty

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/UnCXR_Dataset/checkpoints/seed10.pt")
    parser.add_argument("--img_dir", type=str, default="/content/images_224")
    parser.add_argument("--data_dir", type=str, default="/content/UnX/data")
    parser.add_argument("--preds_dir", type=str, default="/content/drive/MyDrive/UnCXR_Dataset/preds")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.preds_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = build_model(num_classes=len(DISEASES)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint -> {args.checkpoint}")

    for split in ["val", "test"]:
        csv_file = os.path.join(args.data_dir, f"{split}.csv")
        dataset = CXRDataset(csv_file=csv_file, img_dir=args.img_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        print(f"\nRunning MC Dropout on {split} set ({len(dataset)} samples)...")
        mean_probs, uncertainty = run_mc_dropout(model, loader, device, num_samples=args.samples)

        mean_path = os.path.join(args.preds_dir, f"mc_seed10_{split}_mean_probs.npy")
        var_path = os.path.join(args.preds_dir, f"mc_seed10_{split}_var_uncertainty.npy")

        np.save(mean_path, mean_probs)
        np.save(var_path, uncertainty)
        print(f"Saved: {mean_path}")
        print(f"Saved: {var_path}")

if __name__ == "__main__":
    main()
