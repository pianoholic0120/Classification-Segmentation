import os
import argparse
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

def load_model(model_path, device):
    model = DeepLabV3WithResNet101(num_classes=7)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

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

def inference(image_dir, model, save_dir, device):
    image_names = [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
    os.makedirs(save_dir, exist_ok=True)
    
    for img_name in image_names:
        image = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        input_image = transform_img(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_image)
            pred = torch.argmax(output, dim=1)

        save_prediction(pred, save_dir, [img_name])
        print(f"Saved predicted mask for {img_name}")

# Main function (remains the same)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    
    inference(args.image_dir, model, args.save_dir, device)