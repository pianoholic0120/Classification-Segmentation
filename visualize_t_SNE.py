import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.models import resnet50
from officehome import get_dataloaders, Classifier
import os

# Path to save t-SNE visualization
output_path = './visualization_t_SNE.png'

# Load model and dataset (setting C)
def load_model_and_data(train_csv, train_dir, batch_size=64):
    # Load the ResNet50 backbone
    backbone = resnet50(weights=None)
    backbone.fc = nn.Identity()  # Ensure the FC layer is removed
    
    # Create a classifier model
    num_classes = 65  # Number of classes in Office-Home dataset
    model = Classifier(backbone, num_classes=num_classes)
    
    # Load the training data
    train_loader, _ = get_dataloaders(train_csv, "./hw1_data/p1_data/office/val.csv", train_dir, "./hw1_data/p1_data/office/val/", batch_size)
    
    return model, train_loader

# Extract second-to-last layer features
def extract_features(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    features = []
    labels = []
    
    with torch.no_grad():
        for images, label, _ in dataloader:  # Ignore filename by adding underscore _
            images = images.to(device)
            label = label.to(device)
            
            # Forward pass through the model
            feature = model.backbone(images)  # Get output from the backbone (second-last layer)
            features.append(feature.cpu())
            labels.append(label.cpu())
    
    # Concatenate all features and labels
    features = torch.cat(features)
    labels = torch.cat(labels)
    
    return features, labels

# Visualize the t-SNE representation
def visualize_tsne(features, labels, epoch, ax, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', alpha=0.6)
    ax.set_title(f"{title} (Epoch {epoch})")
    ax.legend(*scatter.legend_elements(), title="Classes", loc="best")

# Main function to perform t-SNE visualization for first and last epochs
def tsne_visualization(train_csv, train_dir, checkpoint_first_epoch, checkpoint_last_epoch, batch_size=64, output_path='./visualization_t_SNE.png'):
    # Load the training data
    model, train_loader = load_model_and_data(train_csv, train_dir, batch_size)
    
    # Device configuration (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create subplots for the t-SNE visualization (First and Last Epoch)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # First Epoch: Load checkpoint and extract features
    checkpoint_first = torch.load(checkpoint_first_epoch, map_location=device)
    model.load_state_dict(checkpoint_first['model_state_dict'])
    features_first, labels_first = extract_features(model, train_loader, device)
    
    # Visualize t-SNE for the first epoch
    visualize_tsne(features_first.numpy(), labels_first.numpy(), epoch=checkpoint_first['epoch'], ax=ax[0], title="t-SNE at First Epoch")
    
    # Last Epoch: Load checkpoint and extract features
    checkpoint_last = torch.load(checkpoint_last_epoch, map_location=device)
    model.load_state_dict(checkpoint_last['model_state_dict'])
    features_last, labels_last = extract_features(model, train_loader, device)
    
    # Visualize t-SNE for the last epoch
    visualize_tsne(features_last.numpy(), labels_last.numpy(), epoch=checkpoint_last['epoch'], ax=ax[1], title="t-SNE at Last Epoch")
    
    # Save the t-SNE visualization
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"t-SNE visualization saved at {output_path}")
    
# Example usage (Replace with actual paths)
if __name__ == "__main__":
    train_csv = "./hw1_data/p1_data/office/train.csv"
    train_dir = "./hw1_data/p1_data/office/train"
    checkpoint_first_epoch = "./result_problem1/checkpoint_C_0.4828/first_finetuned_model.pth"
    checkpoint_last_epoch = "./result_problem1/checkpoint_C_0.4828/best_finetuned_model.pth"
    
    tsne_visualization(train_csv, train_dir, checkpoint_first_epoch, checkpoint_last_epoch)
