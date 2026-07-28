#!/usr/bin/env python3
"""
Mask R-CNN Training Script for Sudoku Grid Segmentation

This script trains a Mask R-CNN model for segmenting Sudoku grids in images.
It uses YOLOv8 segmentation format for the dataset.
"""

import os
import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import json
from datetime import datetime


def load_yolov8_segmentation(label_path, img_size):
    """
    Load YOLOv8 segmentation format labels and convert to Mask R-CNN format.
    
    Args:
        label_path (str): Path to the label file
        img_size (tuple): Image dimensions (height, width)
    
    Returns:
        tuple: (boxes, labels, masks) tensors
    """
    H, W = img_size
    boxes, labels, masks = [], [], []

    if not os.path.exists(label_path):
        return torch.zeros((0, 4), dtype=torch.float32), \
               torch.zeros((0,), dtype=torch.int64), \
               torch.zeros((0, H, W), dtype=torch.uint8)

    with open(label_path, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            if len(data) < 7:
                continue

            cls_id = int(data[0])
            polygon = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
            polygon[:, 0] *= W
            polygon[:, 1] *= H

            if polygon.shape[0] < 3:
                continue

            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
            if mask.sum() == 0:
                continue

            x_min = np.min(polygon[:, 0])
            y_min = np.min(polygon[:, 1])
            x_max = np.max(polygon[:, 0])
            y_max = np.max(polygon[:, 1])
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(cls_id + 1)  # +1 because background is class 0
            masks.append(mask)

    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), \
               torch.zeros((0,), dtype=torch.int64), \
               torch.zeros((0, H, W), dtype=torch.uint8)

    return torch.tensor(boxes, dtype=torch.float32), \
           torch.tensor(labels, dtype=torch.int64), \
           torch.tensor(np.array(masks), dtype=torch.uint8)


class YOLOv8SegmentationDataset(torch.utils.data.Dataset):
    """Dataset class for YOLOv8 segmentation format."""
    
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.images = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        while True:
            img_path = self.images[idx]
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            label_path = self.label_dir / (img_path.stem + ".txt")
            boxes, labels, masks = load_yolov8_segmentation(label_path, (H, W))
            if boxes.shape[0] == 0:
                idx = np.random.randint(0, len(self.images))
                continue

            target = {"boxes": boxes, "labels": labels, "masks": masks}
            if self.transforms:
                img, target = self.transforms(img, target)
            return torchvision.transforms.ToTensor()(img), target


def get_instance_segmentation_model(num_classes):
    """
    Create a Mask R-CNN model for instance segmentation.
    
    Args:
        num_classes (int): Number of classes (including background)
    
    Returns:
        torch.nn.Module: Mask R-CNN model
    """
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def train_model(model, train_loader, val_loader, optimizer, device, 
                num_epochs=10, early_stop_delta=1e-3, patience=3, max_grad_norm=5.0):
    """
    Train the Mask R-CNN model with early stopping.
    
    Args:
        model: Mask R-CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        early_stop_delta: Minimum improvement for early stopping
        patience: Number of epochs to wait before early stopping
        max_grad_norm: Maximum gradient norm for clipping
    """
    best_loss = float('inf')
    epochs_no_improve = 0
    training_history = []

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip batch if empty targets exist
            if any(len(t["boxes"]) == 0 for t in targets):
                continue

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                if len(targets) == 0 or all(len(t['boxes']) == 0 for t in targets):
                    continue  # skip empty targets to avoid list output

                # Ensure model returns dict
                model.train()  # temporarily compute losses
                loss_dict = model(images, targets)
                if isinstance(loss_dict, list):
                    # happens if targets are empty, skip
                    continue
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                model.eval()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Record training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        # ---- Early Stopping Check ----
        if best_loss - avg_val_loss > early_stop_delta:
            best_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return training_history


def save_model(model, save_path, training_history=None):
    """Save the trained model and optionally training history."""
    # Save model state dict
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Save training history if provided
    if training_history:
        history_path = save_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for Sudoku segmentation')
    parser.add_argument('--data_root', type=str, 
                       default='../../data/segmentation/segmentation_dataset',
                       help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='/home/viny/CV-sudoku-solver/models/weights', 
                       help='Directory to save trained models')
    parser.add_argument('--model_name', type=str, default='maskrcnn_sudoku', 
                       help='Name for the saved model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths
    train_image_dir = os.path.join(args.data_root, 'train', 'images')
    train_label_dir = os.path.join(args.data_root, 'train', 'labels')
    val_image_dir = os.path.join(args.data_root, 'valid', 'images')
    val_label_dir = os.path.join(args.data_root, 'valid', 'labels')
    
    print(f"Training images: {train_image_dir}")
    print(f"Training labels: {train_label_dir}")
    print(f"Validation images: {val_image_dir}")
    print(f"Validation labels: {val_label_dir}")
    
    # Create datasets
    train_dataset = YOLOv8SegmentationDataset(train_image_dir, train_label_dir)
    val_dataset = YOLOv8SegmentationDataset(val_image_dir, val_label_dir)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=lambda x: tuple(zip(*x)))
    
    # Create model
    num_classes = 2  # background + sudoku
    model = get_instance_segmentation_model(num_classes)
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, 
                               weight_decay=args.weight_decay)
    
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Train model
    training_history = train_model(
        model, train_loader, val_loader, optimizer, device, 
        num_epochs=args.num_epochs, patience=args.patience
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"{args.model_name}_{timestamp}.pth")
    save_model(model, model_path, training_history)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
