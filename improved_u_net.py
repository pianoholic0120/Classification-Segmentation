import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
import imageio
from tqdm import tqdm
import torch.nn.functional as F

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

# Define RGB values to class mappings
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

def lovasz_softmax_flat(probas, labels, classes='present', epsilon=1e-6):
    losses = []
    for c in range(probas.size(1)):
        fg = (labels == c).float()  # foreground for class c
        if fg.sum() == 0:
            losses.append(torch.tensor(epsilon).to(probas.device))
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
class DeepLabV3WithResNet101(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(DeepLabV3WithResNet101, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        outputs = self.model(x)['out']
        return outputs

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_train=None, transform_val=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
        self.transform_train = transform_train
        self.transform_val = transform_val

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

        if self.transform_train:
            image = self.transform_train(image)

        # No need to transform the mask to tensor again if it's already a tensor
        if self.transform_val:
            mask = self.transform_val(mask) if not isinstance(mask, torch.Tensor) else mask

        return image, mask

    @staticmethod
    def rgb_to_class_index(mask_rgb):
        mask_rgb = np.array(mask_rgb)
        mask_class = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)

        for rgb_value, class_index in class_rgb_values.items():
            mask_class[(mask_rgb == rgb_value).all(axis=2)] = class_index

        return mask_class

# Save predicted mask
def save_prediction(predictions, save_dir, image_names):
    os.makedirs(save_dir, exist_ok=True)
    predictions = predictions.cpu().numpy()
    for i, pred in enumerate(predictions):
        pred_mask = np.zeros((512, 512, 3), dtype=np.uint8)
        for rgb_value, class_index in class_rgb_values.items():
            pred_mask[(pred == class_index)] = rgb_value
        imageio.imwrite(os.path.join(save_dir, image_names[i].replace('_sat.jpg', '_mask.png')), pred_mask)

# Focal Loss
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
    
# Training function with dynamic loss weighting
def fine_tune_model(model, dataloaders, optimizer, scheduler, criterion, num_epochs=100, device="cuda", save_dir="./mask_pred_trial"):
    best_miou = 0.0
    dice_loss = DiceLoss()
    focal_loss = FocalLoss()
    
    # Add Lovász-Softmax loss
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

                    # CrossEntropy loss
                    loss_ce = criterion(outputs, masks)
                    
                    # Focal and Dice loss
                    loss_focal = focal_loss(outputs, masks)
                    loss_dice = dice_loss(outputs, masks)
                    
                    # Lovász-Softmax loss (on probability maps)
                    probas = F.softmax(outputs, dim=1)
                    loss_lovasz = lovasz_loss(probas, masks)

                    # 動態調整 alpha_focal 和 alpha_dice
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

                # 定義 val_metrics 並將 mIoU 作為 metric
                val_metrics = {'mIoU': epoch_miou, 'iou_list': iou_list}

                # 保存最佳模型
                if epoch_miou > best_miou:
                    best_miou = epoch_miou
                    torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            if phase == 'val':
                scheduler.step(epoch_miou)

    print(f"Best val mIoU: {best_miou:.4f}")



# Calculate class weights function (remains the same)
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

# Main function (remains the same)
if __name__ == "__main__":
    num_classes = 7
    batch_size = 4
    num_epochs_finetune = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_image_dir = './hw1_data/p2_data/train/'
    val_image_dir = './hw1_data/p2_data/validation/'

    train_dataset = SegmentationDataset(train_image_dir, train_image_dir, transform_train=transform_img, transform_val=transform_mask)
    val_dataset = SegmentationDataset(val_image_dir, val_image_dir, transform_train=transform_img, transform_val=transform_mask)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }

    model = DeepLabV3WithResNet101(num_classes=num_classes).to(device)

    class_weights = calculate_class_weights(train_dataset, device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adjust learning rate and optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)  # Decrease learning rate

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    fine_tune_model(model, dataloaders, optimizer, scheduler, criterion, num_epochs=num_epochs_finetune, device=device)