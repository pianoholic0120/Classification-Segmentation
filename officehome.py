import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

# Dataset for Office-Home
class OfficeHomeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)
        
        # Return image, label, and filename
        filename = os.path.basename(img_name)  # Extract only the filename
        return image, label, filename

# Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders(train_csv=None, val_csv=None, train_dir=None, val_dir=None, batch_size=64):
    # If training data is provided, create the training dataset
    if train_csv and train_dir:
        train_dataset = OfficeHomeDataset(train_csv, train_dir, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None

    # Validation dataset is required
    if val_csv and val_dir:
        val_dataset = OfficeHomeDataset(val_csv, val_dir, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Validation CSV and directory are required for validation.")

    return train_loader, val_loader


# Classifier
class Classifier(nn.Module):
    def __init__(self, backbone, num_classes=65, freeze_backbone=False):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(2048, num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        output = self.fc(features)
        return output

# Training and validation functions
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels, _ in tqdm(dataloader):  # Ignore the filename
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader):  # Ignore the filename
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
    return running_loss / len(dataloader.dataset), correct.double() / len(dataloader.dataset)


# Save model
# Save the best model based on validation accuracy
def save_checkpoint(model, optimizer, epoch, best_val_acc, path="best_finetuned_model.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }
    torch.save(checkpoint, path)
    print(f"Best model saved at epoch {epoch} with validation accuracy {best_val_acc:.4f}")

# Main function to train and validate model
def run_training(backbone, train_loader, val_loader, num_classes=65, epochs=10, lr=0.001, freeze_backbone=False, save_path="best_finetuned_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create classifier with the loaded backbone
    model = Classifier(backbone, num_classes=num_classes, freeze_backbone=freeze_backbone).to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0  # Track the best validation accuracy

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Save the model if it has the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, save_path)

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
