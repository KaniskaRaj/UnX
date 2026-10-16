import os
import torch
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Pneumonia", "Pneumothorax"
]

def get_cxr_transforms():
    weights = models.DenseNet121_Weights.DEFAULT
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    ])

class FastCXRDataset(Dataset):
    def __init__(self, df_or_csv, img_dir, transform=None):
        if isinstance(df_or_csv, str):
            self.df = pd.read_csv(df_or_csv).reset_index(drop=True)
        else:
            self.df = df_or_csv.reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform if transform is not None else get_cxr_transforms()
        self.labels = self.df[DISEASES].to_numpy(dtype="float32")
        self.filenames = self.df["Image Index"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        with Image.open(path) as img:
            img_t = self.transform(img)
        return img_t, torch.tensor(self.labels[idx], dtype=torch.float32)
