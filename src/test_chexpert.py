import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 
    'Infiltration', 'Mass', 'Nodule', 'Pneumothorax'
]

class CheXpertDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if 'full_path' in self.df.columns and os.path.exists(str(row['full_path'])):
            img_path = str(row['full_path'])
        else:
            cand = os.path.join(self.img_dir, str(row['Path']))
            if os.path.exists(cand):
                img_path = cand
            else:
                parts = str(row['Path']).split('/', 1)
                img_path = os.path.join(self.img_dir, parts[1]) if len(parts) > 1 else cand

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--img_dir', type=str, default='/content/chexpert_raw')
    parser.add_argument('--preds_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mc_passes', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.preds_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CheXpertDataset(args.csv_path, args.img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Total CheXpert OOD images to evaluate: {len(dataset)}")

    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1024, len(DISEASE_CLASSES))
    )
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    print("Extracting features from backbone...")
    features_list = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Backbone passes"):
            imgs = imgs.to(device)
            feats = model.features(imgs)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = torch.flatten(feats, 1)
            features_list.append(feats.cpu())

    all_features = torch.cat(features_list, dim=0)
    num_samples = all_features.shape[0]

    print("Computing deterministic baseline logits...")
    model.classifier[0].eval()
    with torch.no_grad():
        baseline_logits = model.classifier(all_features.to(device)).cpu().numpy()

    print(f"Executing {args.mc_passes} MC Dropout draws on classification head...")
    model.classifier[0].train()
    mc_preds = np.zeros((args.mc_passes, num_samples, len(DISEASE_CLASSES)), dtype=np.float32)

    feat_loader = DataLoader(all_features, batch_size=256, shuffle=False)
    with torch.no_grad():
        for pass_idx in tqdm(range(args.mc_passes), desc="MC Draws"):
            draw_probs = []
            for batch_feats in feat_loader:
                batch_feats = batch_feats.to(device)
                logits = model.classifier(batch_feats)
                probs = torch.sigmoid(logits).cpu().numpy()
                draw_probs.append(probs)
            mc_preds[pass_idx] = np.concatenate(draw_probs, axis=0)

    out_logits_path = os.path.join(args.preds_dir, "chexpert_seed10_logits.npy")
    out_mc_path = os.path.join(args.preds_dir, "mc_seed10_chexpert.npy")

    np.save(out_logits_path, baseline_logits)
    np.save(out_mc_path, mc_preds)

    print(f"\n✅ Finished! Deliverables saved:")
    print(f"   - {out_logits_path} (Shape: {baseline_logits.shape})")
    print(f"   - {out_mc_path} (Shape: {mc_preds.shape})")

if __name__ == '__main__':
    main()
