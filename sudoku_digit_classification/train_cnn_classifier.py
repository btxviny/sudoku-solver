import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from loguru import logger

from models.model import DigitAndHandwrittenCNN

# Dataset Class
class DigitDataset(Dataset):
    def __init__(self, digit_labels_path, handwritten_labels_path, images_root_dir):
        with open(digit_labels_path, 'r') as f1, open(handwritten_labels_path, 'r') as f2:
            self.digit_labels = f1.read().splitlines(False)
            self.handwritten_labels = f2.read().splitlines(False)
        self.images_paths = [os.path.join(images_root_dir, row.split(',')[0]) for row in self.digit_labels]
    
    def __len__(self):
        return len(self.images_paths)
    
    def __one_hot_encode(self, label, num_classes):
        return torch.eye(num_classes)[label]
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = torch.from_numpy(image / 255.0).float().unsqueeze(0).repeat(3, 1, 1)  # Convert to 3-channel
        
        digit_label = int(self.digit_labels[idx].split(',')[1]) - 1
        digit_label = self.__one_hot_encode(digit_label, 9)

        handwritten_label = self.handwritten_labels[idx].split(',')[1]
        handwritten_label = self.__one_hot_encode(int(handwritten_label), 2)

        return image, digit_label, handwritten_label


# Training Function
def train(model, train_loader, optimizer, digit_criterion, handwritten_criterion, device, writer, epoch):
    model.train()
    running_digit_loss, running_handwritten_loss, running_total_loss = 0.0, 0.0, 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    for batch_idx, (images, digit_labels, handwritten_labels) in enumerate(progress_bar):
        images, digit_labels, handwritten_labels = images.to(device), digit_labels.to(device), handwritten_labels.to(device)
        
        optimizer.zero_grad()
        digit_pred, handwritten_pred = model(images)

        loss_digit = digit_criterion(digit_pred, digit_labels)
        loss_handwritten = handwritten_criterion(handwritten_pred, handwritten_labels)
        total_loss = loss_digit + loss_handwritten

        total_loss.backward()
        optimizer.step()

        running_digit_loss += loss_digit.item()
        running_handwritten_loss += loss_handwritten.item()
        running_total_loss += total_loss.item()

        progress_bar.set_postfix({
            'Digit Loss': f'{running_digit_loss/(batch_idx+1):.4f}',
            'Handwritten Loss': f'{running_handwritten_loss/(batch_idx+1):.4f}',
            'Total Loss': f'{running_total_loss/(batch_idx+1):.4f}'
        })

    return running_total_loss / len(train_loader)


# Validation Function
def validate(model, eval_loader, device):
    model.eval()
    running_digit_correct, running_handwritten_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, digit_labels, handwritten_labels in tqdm(eval_loader, desc="Validation", leave=False):
            images, digit_labels, handwritten_labels = images.to(device), digit_labels.to(device), handwritten_labels.to(device)
            
            digit_pred, handwritten_pred = model(images)

            _, predicted_digits = torch.max(digit_pred, 1)
            _, expected_digits = torch.max(digit_labels, 1)
            running_digit_correct += (predicted_digits == expected_digits).sum().item()

            _, predicted_handwritten = torch.max(handwritten_pred, 1)
            _, expected_handwritten = torch.max(handwritten_labels, 1)
            running_handwritten_correct += (predicted_handwritten == expected_handwritten).sum().item()
            
            total_samples += images.size(0)

    return running_digit_correct / total_samples, running_handwritten_correct / total_samples


# Main Function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Creating Dataset and DataLoader")
    dataset = DigitDataset(args.digit_labels, args.handwritten_labels, args.images_dir)
    train_size = int(len(dataset) * 0.7)
    eval_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - eval_size

    train_dataset, eval_dataset, _ = random_split(dataset, [train_size, eval_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("Instantiating Network")
    model = DigitAndHandwrittenCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    digit_criterion = nn.CrossEntropyLoss()
    handwritten_criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('runs/digit_handwritten_classifier')

    # Learning Rate Scheduler (Reduce LR on Plateau)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Running Training Loop")
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, digit_criterion, handwritten_criterion, device, writer, epoch)
        digit_acc, handwritten_acc = validate(model, eval_loader, device)

        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Digit Acc: {digit_acc:.4f}, Handwritten Acc: {handwritten_acc:.4f}")

        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Accuracy/Digit', digit_acc, epoch)
        writer.add_scalar('Accuracy/Handwritten', handwritten_acc, epoch)

        # Step the scheduler
        lr_scheduler.step(train_loss)

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))

    writer.close()


if __name__ == "__main__":

    digit_labels_path = 'digits/handwritten_and_digital/labels.txt'
    handwritten_labels_path = 'digits/handwritten_and_digital/handwritten_labels.txt'
    images_root_dir = 'digits/handwritten_and_digital/images'
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit_labels", type=str, default=digit_labels_path)
    parser.add_argument("--handwritten_labels", type=str, default=handwritten_labels_path)
    parser.add_argument("--images_dir", type=str, default=images_root_dir)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    main(args)
