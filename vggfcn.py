import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import VGG16_Weights, vgg16
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import imageio
from tqdm import tqdm  # Add tqdm for progress tracking

# Define the height and width
HEIGHT = 512
WIDTH = 512

# Image and mask transformations
transform_img = T.Compose([
    T.Resize([HEIGHT, WIDTH]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mask = T.Compose([
    T.Resize([HEIGHT, WIDTH], interpolation=Image.NEAREST),
    T.ToTensor()
])

def save_prediction(predictions, save_dir, image_names):
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = predictions.cpu().numpy()  # Convert to numpy array
    
    for i, pred in enumerate(predictions):
        # Convert class index mask to RGB (optional) or save as grayscale (required format)
        pred_mask = np.zeros((512, 512, 3), dtype=np.uint8)
        
        for rgb_value, class_index in class_rgb_values.items():
            pred_mask[(pred == class_index)] = rgb_value
        
        # Save the mask as .png
        imageio.imwrite(os.path.join(save_dir, image_names[i].replace('_sat.jpg', '_mask.png')), pred_mask)

# Class mappings for RGB values
class_rgb_values = {
    (0, 255, 255): 0,    # Urban
    (255, 255, 0): 1,    # Agriculture
    (255, 0, 255): 2,    # Rangeland
    (0, 255, 0): 3,      # Forest
    (0, 0, 255): 4,      # Water
    (255, 255, 255): 5,  # Barren
    (0, 0, 0): 6         # Unknown
}
# Lovasz-Softmax Loss
def lovasz_softmax(probs, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        loss = torch.mean(torch.stack([lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lbl.unsqueeze(0), ignore))
                    for prob, lbl in zip(probs, labels)]))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probs, labels, ignore), classes=classes)
    return loss

def lovasz_softmax_flat(probas, labels, classes='present'):
    if classes == 'present':
        class_to_sum = list(range(probas.size(1)))
    losses = []
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.mean(torch.stack(losses))

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch and removes the labels which should be ignored.
    probas: [B, C, H, W] tensor, class probabilities
    labels: [B, H, W] tensor, ground truth labels
    ignore: int, ignore class label (optional)
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
    labels = labels.view(-1)  # (B*H*W)
    if ignore is not None:
        mask = labels != ignore
        probas = probas[mask]
        labels = labels[mask]
    return probas, labels

def compute_dynamic_loss_weights(val_metrics):
    class_ious = val_metrics['iou_list']
    alpha = [0.25 if iou >= 0.5 else 1 for iou in class_ious]
    return alpha

# Define the custom dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("RGB")

        # Convert mask from RGB to class indices
        mask = self.rgb_to_class_index(mask)
        mask = torch.from_numpy(mask).long()

        if self.transform_img:
            image = self.transform_img(image)

        # Apply transform_mask only if the mask is not already a tensor
        if isinstance(mask, Image.Image):  # Check if it's still an image
            mask = self.transform_mask(mask)

        return image, mask

    @staticmethod
    def rgb_to_class_index(mask_rgb):
        mask_rgb = np.array(mask_rgb)
        mask_class = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
        for rgb_value, class_index in class_rgb_values.items():
            mask_class[(mask_rgb == rgb_value).all(axis=2)] = class_index
        return mask_class

# Define the FCN32s model with VGG-16 backbone
class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()

        # Use the pretrained VGG16 model
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features  # Use VGG16 feature layers

        # FC layers as 1x1 convolutions
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7)  # Conv6
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.7)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)  # Conv7
        
        # Final scoring layer
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Upsampling layer
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)

        # Initialize weights from VGG16
        self.init_vgg16_params(vgg16)

    def forward(self, x):
        # Pass through VGG16-like layers
        x = self.features(x)
        
        # Fully connected layers
        x = self.relu(self.conv6(x))  # Conv6
        x = self.dropout(x)
        x = self.relu(self.conv7(x))  # Conv7
        x = self.dropout(x)

        # Score
        x = self.score_fr(x)

        # Upsample to the original size
        x = self.upscore(x)

        # Resize output to 512x512 using bilinear interpolation
        x = nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        return x

    def init_vgg16_params(self, vgg16):
        """
        Initialize the FCN32s model with parameters from the pretrained VGG16 model.
        """
        # Copy parameters from the VGG16 feature extractor layers
        for l1, l2 in zip(vgg16.features, self.features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=7):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes)  
        else:
            self.alpha = torch.tensor(alpha)  
        self.gamma = gamma

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)

        ce_loss = nn.CrossEntropyLoss(ignore_index=6)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss * self.alpha
        return focal_loss


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        num_classes = preds.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) + self.smooth)
        return 1 - dice.mean()
    
# Training function with tqdm for progress bars
def train_model(model, dataloaders, optimizer, scheduler, criterion, num_epochs=200, device="cuda", save_dir="./mask_pred_vggfcn"):
    best_miou = 0.0
    dice_loss = DiceLoss()
    focal_loss = FocalLoss()
    lovasz_loss = lovasz_softmax
    alpha_focal, alpha_dice = 0.4, 0.6
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_masks = []
            all_names = []

            dataloader_tqdm = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase", leave=False)

            for inputs, masks in dataloader_tqdm:
                inputs = inputs.to(device)
                masks = masks.to(device)
                image_names = [img_name for img_name in dataloaders[phase].dataset.image_names]
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)

                    loss_ce = criterion(outputs, masks)
                    # Focal and Dice loss
                    loss_focal = focal_loss(outputs, masks)
                    loss_dice = dice_loss(outputs, masks)
                    
                    # Lovász-Softmax loss (on probability maps)
                    probas = F.softmax(outputs, dim=1)
                    loss_lovasz = lovasz_loss(probas, masks)
                    if phase == 'val' and epoch > 0:  # 確保在驗證階段和有過一次訓練後調整
                        focal_loss.alpha= torch.tensor(compute_dynamic_loss_weights(val_metrics))
                    
                    loss = loss_ce + alpha_focal * loss_focal + alpha_dice * loss_dice + loss_lovasz

                    if loss.dim() > 0:
                        loss = loss.mean()
                        
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                if phase == 'val':
                    all_preds.append(preds.cpu())
                    all_masks.append(masks.cpu())
                    all_names.extend(image_names)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

            if phase == 'val':
                all_preds = torch.cat(all_preds, dim=0)
                save_prediction(all_preds, save_dir, all_names)
                all_masks = torch.cat(all_masks, dim=0)

                # New mIoU calculation
                iou_list = []
                num_classes = 6  # Exclude Unknown class (6)
                for class_index in range(num_classes):
                    tp = ((all_preds == class_index) & (all_masks == class_index)).sum().item()  # True Positive
                    fp = ((all_preds == class_index) & (all_masks != class_index)).sum().item()  # False Positive
                    fn = ((all_preds != class_index) & (all_masks == class_index)).sum().item()  # False Negative
                    
                    # Calculate IoU
                    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                    iou_list.append(iou)

                epoch_miou = sum(iou_list) / len(iou_list)
                print(f"Validation mIoU: {epoch_miou:.4f}")
                val_metrics = {'mIoU': epoch_miou, 'iou_list': iou_list}

                # 保存最佳模型
                if epoch_miou > best_miou:
                    best_miou = epoch_miou
                    torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            if phase == 'val':
                scheduler.step(epoch_miou)

    print(f"Best val mIoU: {best_miou:.4f}")

def calculate_class_weights(dataset, device, num_classes=7, normalize=True):
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    
    for _, mask in dataset:
        # Count the number of pixels per class
        for class_id in range(num_classes):
            class_counts[class_id] += (mask == class_id).sum().item()
    
    total_samples = class_counts.sum().item()
    
    # Avoid division by zero for classes that have no samples
    class_weights = total_samples / (num_classes * (class_counts + 1e-6))
    
    # Optional normalization of weights
    if normalize:
        class_weights = class_weights / class_weights.sum()
    
    return class_weights.to(device)

# Main code
if __name__ == "__main__":
    num_classes = 7
    batch_size = 4
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset directories
    train_image_dir = './hw1_data/p2_data/train/'
    val_image_dir = './hw1_data/p2_data/validation/'

    # Create datasets and dataloaders
    train_dataset = SegmentationDataset(train_image_dir, train_image_dir, transform_img=transform_img, transform_mask=transform_mask)
    val_dataset = SegmentationDataset(val_image_dir, val_image_dir, transform_img=transform_img, transform_mask=transform_mask)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }

    # Initialize model, loss, optimizer, and scheduler
    model = FCN32s(num_classes=num_classes).to(device)
    class_weights = calculate_class_weights(train_dataset, device=device)
    # class_weights[2] *= 2.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Adjust learning rate and optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)  # Decrease learning rate

    # Use ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Train the model
    train_model(model, dataloaders, optimizer, scheduler, criterion, num_epochs=num_epochs, device=device)
