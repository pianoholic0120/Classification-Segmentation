import argparse
from torchvision.models import resnet50
import torch
import torch.nn as nn
from officehome import get_dataloaders, run_training, validate_epoch, Classifier
import os
import csv

# Paths for provided weights (TA's backbone) and SSL pre-trained weights
TA_BACKBONE_PATH = './hw1_data/p1_data/pretrain_model_SL.pt'
SSL_BACKBONE_PATH = './output/ssp_test1/best_checkpoint.pth'

# Load backbone based on setting
def load_backbone(setting, pretrained=False):
    if setting in ['B', 'D']:  # TA's provided backbone
        backbone = resnet50(weights=None)
        backbone.load_state_dict(torch.load(TA_BACKBONE_PATH))
    elif setting in ['C', 'E']:  # SSL pre-trained backbone
        backbone = resnet50(weights=None)
        checkpoint = torch.load(SSL_BACKBONE_PATH)
        
        # Load state dict with strict=False to handle mismatch in BN layers
        model_state_dict = checkpoint['model_state_dict']
        missing, unexpected = backbone.load_state_dict(model_state_dict, strict=False)
        print(f"Missing keys when loading backbone: {missing}")
        print(f"Unexpected keys when loading backbone: {unexpected}")
    else:  # Random initialization (case A)
        backbone = resnet50(pretrained=pretrained)
    
    # Remove fully connected layer
    backbone.fc = nn.Identity()
    return backbone

# Experiment function to fine-tune and save the best model
def experiment(setting, train_csv, val_csv, train_dir, val_dir, batch_size=64, epochs=10, lr=0.001, save_path="best_finetuned_model.pth"):
    # Load the ResNet50 backbone (either TA's or SSL pre-trained)
    backbone = load_backbone(setting)
    
    # Determine if backbone should be frozen (for settings D, E)
    freeze_backbone = True if setting in ['D', 'E'] else False
    
    # Load the training and validation datasets
    train_loader, val_loader = get_dataloaders(train_csv, val_csv, train_dir, val_dir, batch_size)
    
    # Run training and validation with fine-tuning
    run_training(backbone, train_loader, val_loader, epochs=epochs, lr=lr, freeze_backbone=freeze_backbone, save_path=save_path)

# Function to load a fine-tuned model and test its accuracy on the validation set
def test_model(checkpoint_path, val_csv, val_dir, batch_size=64, output_csv="predictions.csv"):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint file '{checkpoint_path}' does not exist.")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Load the ResNet50 backbone
    backbone = resnet50(weights=None)
    backbone.fc = nn.Identity()  # Ensure the FC layer is removed
    
    # Create a classifier model
    num_classes = 65  # Number of classes in Office-Home dataset
    model = Classifier(backbone, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the validation data
    _, val_loader = get_dataloaders(None, val_csv, None, val_dir, batch_size)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    predictions = []
    image_id = 0  # Initialize image_id counter
    
    # Predict labels for all validation images
    with torch.no_grad():
        for images, _, filenames in val_loader:  # Get images and filenames
            images = images.to(device)
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, 1)
            
            # Store predictions along with filenames and image_id
            for idx, filename in enumerate(filenames):
                label = predicted_labels[idx].item()
                predictions.append((image_id, filename, label))
                image_id += 1  # Increment image_id for each prediction
    
    # Write predictions to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'filename', 'label'])  # Write header
        
        # Write each image_id, filename, and predicted label
        for image_id, filename, label in predictions:
            writer.writerow([image_id, filename, label])
    
    print(f"Predictions saved to {output_csv}")

    return predictions

# Main function to parse arguments and run the experiment or test model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune or test a ResNet model for Office-Home dataset")
    
    # Mode argument: 'train' or 'test'
    parser.add_argument('--mode', type=str, required=True, help="Mode: 'train' for fine-tuning, 'test' for testing")
    
    # Arguments for the experiment (training)
    parser.add_argument('--setting', type=str, default="C", help="Training setting (A, B, C, D, or E)")
    parser.add_argument('--train_csv', type=str, default="./hw1_data/p1_data/office/train.csv", help="Path to training CSV file")
    parser.add_argument('--val_csv', type=str, required=True, help="Path to validation CSV file")
    parser.add_argument('--train_dir', type=str, default="./hw1_data/p1_data/office/train", help="Directory path to training images")
    parser.add_argument('--val_dir', type=str, required=True, help="Directory path to validation images")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training or testing")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--save_path', type=str, default="./best_finetuned_model.pth", help="Path to save or load the fine-tuned model")
    parser.add_argument('--output_csv', type=str, default="./predictions.csv")

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'train':
        # Ensure all training arguments are provided
        if not args.setting or not args.train_csv or not args.train_dir:
            raise ValueError("Training requires --setting, --train_csv, and --train_dir arguments.")
        
        # Run the fine-tuning experiment
        experiment(
            setting=args.setting,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.save_path
        )
    
    elif args.mode == 'test':
        # Test the fine-tuned model
        test_model(
            checkpoint_path=args.save_path,
            val_csv=args.val_csv,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            output_csv=args.output_csv
        )
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")
