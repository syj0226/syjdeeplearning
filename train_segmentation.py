"""
DeepLabV3 语义分割训练脚本
数据集: PASCAL VOC 2012 (街景理解)
运行方式: python train_segmentation.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime

# ============ 配置 ============
VOC_PATH = r"C:\Users\Lenovo\Desktop\test\ww\data\VOCdevkit\VOC2012"
BATCH_SIZE = 8  # 8GB显存建议用8，如果显存不够改成4
EPOCHS = 30
LR = 0.001
USE_AMP = True
NUM_CLASSES = 21  # VOC有21个类别（含背景）
IGNORE_INDEX = 255  # VOC中255表示忽略区域


# ============ VOC数据集加载器 ============
class VOCSegmentationDataset(Dataset):
    """PASCAL VOC 2012 语义分割数据集"""

    def __init__(self, root_dir, image_set='train', transform=None, target_transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform

        # 读取图片列表
        split_file = self.root_dir / "ImageSets" / "Segmentation" / f"{image_set}.txt"

        # 检查文件是否存在
        if not split_file.exists():
            raise FileNotFoundError(f"找不到文件: {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        print(f"加载 {image_set} 集: {len(self.image_ids)} 张图片")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 图片路径
        image_path = self.root_dir / "JPEGImages" / f"{image_id}.jpg"
        mask_path = self.root_dir / "SegmentationClass" / f"{image_id}.png"

        # 加载图片
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        # 转换mask为numpy
        mask = np.array(mask)
        # VOC标注中255表示忽略区域
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            # 注意: mask需要保持为Long类型，不做归一化
            mask = self.target_transform(mask.unsqueeze(0)).squeeze(0)

        return image, mask


def get_dataloaders(data_root, batch_size=8):
    """获取数据加载器"""

    # 图片预处理
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # mask预处理（只改变大小，不改变数值）
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    # 训练集和验证集
    train_dataset = VOCSegmentationDataset(
        data_root, image_set='train',
        transform=image_transform, target_transform=target_transform
    )
    val_dataset = VOCSegmentationDataset(
        data_root, image_set='val',
        transform=image_transform, target_transform=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


# ============ DeepLabV3模型 ============
class DeepLabV3Segmenter:
    def __init__(self, num_classes=21, device='cuda'):
        self.device = device

        print("\n📦 加载 DeepLabV3 预训练模型...")
        # 加载预训练模型（在COCO上预训练，包含VOC类别）
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        )

        # 修改分类头，适配VOC的21类
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        self.model = self.model.to(device)

        # 打印参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")

    def train_epoch(self, loader, criterion, optimizer, scaler, epoch, writer, use_amp=True):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f'Epoch {epoch} [分割训练]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = self.model(inputs)['out']  # DeepLabV3返回字典
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / (batch_idx + 1):.4f}'})

            if batch_idx % 20 == 0:
                step = epoch * len(loader) + batch_idx
                writer.add_scalar('Train_Seg/Loss', loss.item(), step)

        return running_loss / len(loader)

    def validate(self, loader, criterion, epoch, writer):
        """验证"""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=f'Epoch {epoch} [分割验证]'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)['out']
                loss = criterion(outputs, targets)
                running_loss += loss.item()

        val_loss = running_loss / len(loader)
        writer.add_scalar('Val_Seg/Loss', val_loss, epoch)

        return val_loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 分割模型已保存: {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"📂 分割模型已加载: {path}")


# ============ 可视化分割结果 ============
def visualize_segmentation(model, loader, device, num_samples=3):
    """可视化分割结果"""
    import matplotlib.pyplot as plt

    model.model.eval()

    # VOC颜色映射
    colors = [
        [0, 0, 0],  # 0: 背景
        [128, 0, 0],  # 1: 航空器
        [0, 128, 0],  # 2: 自行车
        [128, 128, 0],  # 3: 鸟
        [0, 0, 128],  # 4: 船
        [128, 0, 128],  # 5: 瓶子
        [0, 128, 128],  # 6: 公交车
        [128, 128, 128],  # 7: 汽车
        [64, 0, 0],  # 8: 猫
        [192, 0, 0],  # 9: 椅子
        [64, 128, 0],  # 10: 牛
        [192, 128, 0],  # 11: 餐桌
        [64, 0, 128],  # 12: 狗
        [192, 0, 128],  # 13: 马
        [64, 128, 128],  # 14: 摩托车
        [192, 128, 128],  # 15: 人
        [0, 64, 0],  # 16: 盆栽
        [128, 64, 0],  # 17: 羊
        [0, 192, 0],  # 18: 沙发
        [128, 192, 0],  # 19: 火车
        [0, 64, 128]  # 20: 电视
    ]

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, (img, mask) in enumerate(loader):
        if i >= num_samples:
            break

        img = img.to(device)
        with torch.no_grad():
            output = model.model(img)['out']
            pred = output.argmax(dim=1)

        # 转回CPU显示
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        mask_np = mask[0].cpu().numpy()
        pred_np = pred[0].cpu().numpy()

        # 显示原图
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # 显示真实mask
        mask_color = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
        for c in range(NUM_CLASSES):
            mask_color[mask_np == c] = colors[c]
        axes[i, 1].imshow(mask_color)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # 显示预测mask
        pred_color = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
        for c in range(NUM_CLASSES):
            pred_color[pred_np == c] = colors[c]
        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=150)
    plt.show()
    print("📸 分割结果已保存: segmentation_results.png")


# ============ GPU性能测试 ============
def profile_gpu_segmentation(model, loader, device):
    """测试分割任务的GPU性能"""
    print("\n" + "=" * 60)
    print("🚀 分割任务 GPU性能对比: AMP vs FP32")
    print("=" * 60)

    results = {}

    for use_amp in [True, False]:
        model.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        optimizer = optim.AdamW(model.model.parameters(), lr=0.001)

        torch.cuda.reset_peak_memory_stats()

        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)

        # 预热
        for _ in range(5):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model.model(inputs)['out']
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(20):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model.model(inputs)['out']
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3

        mode = 'AMP' if use_amp else 'FP32'
        results[mode] = {'time': elapsed / 20, 'memory': peak_memory}

    print(f"\n⚡ AMP:  {results['AMP']['time'] * 1000:.2f}ms/batch, 显存: {results['AMP']['memory']:.2f}GB")
    print(f"🔋 FP32: {results['FP32']['time'] * 1000:.2f}ms/batch, 显存: {results['FP32']['memory']:.2f}GB")
    print(f"📈 加速比: {results['FP32']['time'] / results['AMP']['time']:.2f}x")

    return results


# ============ 主函数 ============
def main():
    print("=" * 60)
    print("🎯 深度学习期末项目 - DeepLabV3 语义分割训练")
    print("📁 数据集: PASCAL VOC 2012 (街景理解)")
    print("=" * 60)

    # 检查数据路径
    voc_path = Path(VOC_PATH)
    if not voc_path.exists():
        print(f"\n❌ VOC路径不存在: {voc_path}")
        print("请确认路径是否正确")
        return

    print(f"\n✅ VOC路径验证通过: {voc_path}")

    # 检查必要子目录
    required_dirs = ["JPEGImages", "SegmentationClass", "ImageSets/Segmentation"]
    for d in required_dirs:
        if not (voc_path / d).exists():
            print(f"❌ 缺少目录: {d}")
            return

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  计算设备: {device}")

    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

    # 加载数据
    print("\n📂 加载VOC数据...")
    try:
        train_loader, val_loader = get_dataloaders(VOC_PATH, BATCH_SIZE)
    except FileNotFoundError as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 创建模型
    segmenter = DeepLabV3Segmenter(NUM_CLASSES, device)

    # GPU性能测试
    if device.type == 'cuda':
        profile_gpu_segmentation(segmenter, train_loader, device)

    # 训练配置
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.AdamW(segmenter.model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/segmentation_{timestamp}')

    # 训练循环
    best_loss = float('inf')
    print("\n" + "=" * 60)
    print("🚀 开始分割训练")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'=' * 50}")

        train_loss = segmenter.train_epoch(
            train_loader, criterion, optimizer, scaler, epoch, writer, USE_AMP
        )
        val_loss = segmenter.validate(val_loader, criterion, epoch, writer)

        print(f"\n📊 Epoch {epoch} 结果:")
        print(f"   训练 Loss: {train_loss:.4f}")
        print(f"   验证 Loss: {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            segmenter.save('best_segmentation.pth')
            print(f"   ✨ 新的最佳模型! Loss: {best_loss:.4f}")

    writer.close()

    print("\n" + "=" * 60)
    print("✅ 分割训练完成!")
    print("=" * 60)
    print(f"\n🏆 最佳验证 Loss: {best_loss:.4f}")
    print(f"📁 TensorBoard日志: runs/segmentation_{timestamp}")

    # 可视化分割结果
    print("\n📸 生成分割结果可视化...")
    visualize_segmentation(segmenter, val_loader, device, num_samples=3)


if __name__ == "__main__":
    main()
