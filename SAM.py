import os
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image

# 设置设备
device = "cuda"  # 或者 "cpu" 取决于你的环境

# 设置模型参数并加载分割模型
sam_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"

# 加载分割模型
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 初始化掩模生成器
mask_generator = SamAutomaticMaskGenerator(sam)

# 加载并生成掩模
def generate_and_save_masks(image_path, save_dir='./mask_pred_SAM/'):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 读取原始图像
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 生成掩模
    masks = mask_generator.generate(image_np)

    # 处理并保存掩模
    if len(masks) == 0:
        print(f"Warning: No masks generated for {image_path}")
        return

    # 创建一个与原图相同大小的空白图像用于叠加
    overlay_image = np.zeros_like(image_np)

    # 将掩模合并到叠加图像中
    for i, mask in enumerate(masks):
        segmentation = mask['segmentation']
        
        # 生成随机颜色
        color = np.random.randint(0, 255, size=3)  # 生成一个随机颜色

        overlay_image[segmentation] = color  # 赋予随机颜色

    # 将叠加图像与原图合并
    # 创建一个透明的图像，方便后续的合成
    alpha = 0.5  # 透明度
    combined_image = np.clip(image_np * (1 - alpha) + overlay_image * alpha, 0, 255).astype(np.uint8)

    # 保存合并图像
    combined_image_pil = Image.fromarray(combined_image)  # 转换为PIL图像
    mask_file_name = os.path.basename(image_path).replace('_sat.jpg', '_mask_pred.png')
    combined_image_pil.save(os.path.join(save_dir, mask_file_name))
    print(f"Saved mask image to {os.path.join(save_dir, mask_file_name)}")

# 指定图像路径并生成掩模
image_path = './hw1_data/p2_data/validation/0112_sat.jpg'
generate_and_save_masks(image_path)
