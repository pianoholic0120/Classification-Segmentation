import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image
import os

output_dir = './output/ssp_test1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# BYOL-specific layers and functions
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super(MLPHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class BYOL(nn.Module):
    def __init__(self, backbone, projection_dim=256, hidden_dim=4096):
        super(BYOL, self).__init__()
        self.backbone = backbone
        self.projector = MLPHead(2048, hidden_dim, projection_dim)
        
        # Target network initialization with strict=False
        self.target_backbone = resnet50(pretrained=False)
        self.target_backbone.fc = nn.Identity()  # Ensure it has the same structure as backbone
        self.target_backbone.load_state_dict(backbone.state_dict(), strict=False)

        self.target_projector = MLPHead(2048, hidden_dim, projection_dim)
        self.target_projector.load_state_dict(self.projector.state_dict(), strict=False)

    @torch.no_grad()
    def update_target_network(self, tau=0.99, epoch=None, total_epochs=100):
        # Decay tau over epochs
        tau = 1 - (1 - tau) * (1 - epoch / total_epochs)  # Decaying tau
        for param_q, param_k in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            param_k.data = tau * param_k.data + (1 - tau) * param_q.data
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data = tau * param_k.data + (1 - tau) * param_q.data


    def forward(self, x1, x2):
        z1_online = self.projector(self.backbone(x1))
        z2_online = self.projector(self.backbone(x2))

        z1_target = self.target_projector(self.target_backbone(x1))
        z2_target = self.target_projector(self.target_backbone(x2))

        return z1_online, z2_online, z1_target, z2_target

def loss_fn(x, y):
    x = nn.functional.normalize(x, dim=-1, p=2)
    y = nn.functional.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# Dataset and transformations
class MiniImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Data Augmentations (Simulating different views of the same image)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Mini-ImageNet dataset
def get_dataloaders(batch_size=64, num_workers=4):
    dataset = MiniImageNetDataset(root_dir="./hw1_data/p1_data/mini/train", transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

# Save the best checkpoint
def save_best_checkpoint(epoch, model, optimizer, scheduler, best_loss, filename='./output/ssp_test1.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }
    torch.save(checkpoint, filename)
    print(f"Best checkpoint saved at epoch {epoch} with loss {best_loss:.4f}")

# Training function with best model saving
def train_byol(model, dataloader, optimizer, scheduler, epochs=100, device='cuda'):
    model = model.to(device)
    model.train()

    best_loss = float('inf')  # Initialize best loss as infinity

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            x1 = batch.to(device)
            x2 = batch.to(device)
            
            z1_online, z2_online, z1_target, z2_target = model(x1, x2)
            
            loss = (loss_fn(z1_online, z2_target).mean() + loss_fn(z2_online, z1_target).mean()) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.update_target_network(tau=0.99, epoch=epoch, total_epochs=epochs)
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / len(dataloader))
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")  # Track loss

        # Check if this is the best model so far, and save it if so
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_best_checkpoint(epoch, model, optimizer, scheduler, best_loss, os.path.join(output_dir, 'best_checkpoint.pth'))

    # Save final model at the end of training
    save_best_checkpoint(epoch, model, optimizer, scheduler, best_loss, os.path.join(output_dir, 'final_checkpoint.pth'))
    print(f"Final checkpoint saved at epoch {epoch+1}.")

# Main script to set up and run the training
if __name__ == "__main__":
    # Set up ResNet50 backbone
    backbone = resnet50(pretrained=False)
    backbone.fc = nn.Identity()  # Remove the fully connected layer

    # BYOL model
    byol_model = BYOL(backbone=backbone)

    # Optimizer and learning rate scheduler
    optimizer = optim.AdamW(byol_model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00003)

    # Get dataloader
    dataloader = get_dataloaders(batch_size=64)

    # Train BYOL and save only the best checkpoint
    train_byol(byol_model, dataloader, optimizer, scheduler, epochs=100, device='cuda')
