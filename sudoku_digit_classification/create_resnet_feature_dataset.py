import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from loguru import logger
import h5py

# ---------------------------
# Dataset Definition
# ---------------------------
class DigitDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'r') as f:
            self.data = f.read().splitlines(False)
        
        self.images_paths = []
        self.labels = []
        
        for row in self.data:
            image_path, label = row.split(',')
            self.images_paths.append(image_path)
            self.labels.append(int(label))  
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label, self.images_paths[idx]

# ---------------------------
# Image Transformations
# ---------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Main Script
# ---------------------------
def main():
    dataset = DigitDataset('digits/labels.txt', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    num_images = len(dataset)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained ResNet18 and remove classifier
    logger.info("Loading ResNet18 model...")
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove final fc layer
    resnet.to(device)
    resnet.eval()

    # Prepare HDF5 file
    h5_path = "digit_embeddings.h5"
    logger.info(f"Creating HDF5 file at {h5_path}...")
    with h5py.File(h5_path, "w") as f:
        dt = h5py.string_dtype(encoding='utf-8')
        filenames_ds = f.create_dataset("filenames", shape=(num_images,), dtype=dt)
        labels_ds = f.create_dataset("labels", shape=(num_images,), dtype='i')
        features_ds = f.create_dataset("features", shape=(num_images, 512), dtype='f')

        idx = 0
        logger.info("Extracting features and saving to HDF5...")
        with torch.inference_mode():
            for images, lbls, fns in tqdm(dataloader, total=len(dataloader)):
                images = images.to(device)
                features = resnet(images)                  # [batch, 512, 1, 1]
                features = features.view(features.size(0), -1).cpu().numpy()  # flatten
                
                batch_size = len(fns)
                filenames_ds[idx:idx+batch_size] = fns
                labels_ds[idx:idx+batch_size] = lbls
                features_ds[idx:idx+batch_size] = features
                idx += batch_size

    logger.info(f"Finished! Saved {num_images} embeddings to {h5_path}")

if __name__ == "__main__":
    main()
