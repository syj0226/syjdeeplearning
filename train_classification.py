"""
ResNeXt 垃圾分类训练脚本
适配你现有的数据结构: data/Garbage classification/Garbage classification/类别/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

# ============ 配置 ============
# 你的实际数据路径（不改文件夹结构）
DATA_PATH = r"D:\python object\pythonProject\深度学习期末\data\Garbage classification\Garbage classification"
VOC_PATH = r"D:\python object\pythonProject\深度学习期末\data\VOCdevkit"

BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
USE_AMP = True  # 混合精度训练


# ============ 垃圾分类数据集加载器 ============
class GarbageDataset(Dataset):
    """垃圾分类数据集 - 适配你的文件夹结构"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # 获取所有类别文件夹（cardboard, glass, metal, paper, plastic, trash）
        self.classes = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        self.classes.sort()  # 排序保证一致性
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 收集所有图片路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            # 支持多种图片格式
            img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            for ext in img_extensions:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        # 统计信息
        print(f"\n📊 数据集统计:")
        print(f"  总图片数: {len(self.samples)}")
        print(f"  类别数: {len(self.classes)}")
        print(f"  类别: {self.classes}")
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            count = sum(1 for ext in img_extensions for _ in cls_dir.glob(ext))
            print(f"    - {cls}: {count} 张")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法读取图片 {img_path}, 错误: {e}")
            # 返回一个黑色图片作为替代
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_root, batch_size=32, val_split=0.2):
    """获取训练和验证数据加载器"""

    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证数据预处理
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载完整数据集
    full_dataset = GarbageDataset(data_root, transform=train_transform)

    # 划分训练/验证集
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 验证集使用不同的transform（需要特殊处理）
    val_dataset.dataset.transform = val_transform

    # 创建DataLoader（num_workers=0 避免Windows多进程问题）
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, full_dataset.classes


# ============ ResNeXt分类模型 ============
class ResNeXtClassifier:
    def __init__(self, num_classes, device='cuda'):
        self.device = device

        print("\n📦 加载预训练模型...")
        # 加载预训练模型
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)

        # 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False

        # 替换分类头
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # 只解冻分类头
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.model = self.model.to(device)

        # 打印参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   冻结参数: {total_params - trainable_params:,}")
        print("\n📐 模型分类头结构:")
        print(self.model.fc)

    def train_epoch(self, loader, criterion, optimizer, scaler, epoch, writer, use_amp=True):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f'Epoch {epoch} [训练]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            # 混合精度前向传播
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{running_loss / (batch_idx + 1):.4f}', 'acc': f'{acc:.2f}%'})

            # 记录到TensorBoard
            if batch_idx % 20 == 0:
                step = epoch * len(loader) + batch_idx
                writer.add_scalar('Train/Loss', loss.item(), step)
                writer.add_scalar('Train/Accuracy', acc, step)

        return running_loss / len(loader), 100. * correct / total

    def validate(self, loader, criterion, epoch, writer):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=f'Epoch {epoch} [验证]'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / len(loader)
        val_acc = 100. * correct / total

        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)

        return val_loss, val_acc

    def save(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"💾 模型已保存: {path}")

    def load(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"📂 模型已加载: {path}")


# ============ GPU性能测试 ============
def profile_gpu_performance(model, loader, device):
    """测试AMP混合精度 vs FP32的性能对比"""
    print("\n" + "=" * 60)
    print("🚀 GPU性能对比测试: AMP vs FP32")
    print("=" * 60)

    results = {}

    for use_amp in [True, False]:
        model.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.model.parameters(), lr=0.001)

        # 重置显存统计
        torch.cuda.reset_peak_memory_stats()

        # 获取一个batch
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)

        # 预热
        for _ in range(5):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model.model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        start = time.time()

        # 测试20个batch
        for _ in range(20):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model.model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        elapsed = time.time() - start

        # 获取峰值显存
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3

        mode = 'AMP' if use_amp else 'FP32'
        results[mode] = {
            'time': elapsed / 20,
            'memory': peak_memory
        }

    # 打印结果
    print(f"\n⚡ AMP:  {results['AMP']['time'] * 1000:.2f}ms/batch, 显存: {results['AMP']['memory']:.2f}GB")
    print(f"🔋 FP32: {results['FP32']['time'] * 1000:.2f}ms/batch, 显存: {results['FP32']['memory']:.2f}GB")

    speedup = results['FP32']['time'] / results['AMP']['time']
    memory_saved = results['FP32']['memory'] - results['AMP']['memory']

    print(f"\n📈 加速比: {speedup:.2f}x (AMP比FP32快{speedup:.1f}倍)")
    print(f"💾 显存节省: {memory_saved:.2f}GB")

    return results


# ============ 主函数 ============
def main():
    print("=" * 60)
    print("🎯 深度学习期末项目 - ResNeXt垃圾分类训练")
    print("=" * 60)

    # 1. 检查数据路径
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print(f"\n❌ 错误: 数据路径不存在!")
        print(f"   路径: {data_path}")
        print(f"\n请确认你的数据存放在:")
        print(f"   {DATA_PATH}")
        print(f"\n期望的目录结构:")
        print(f"   data/Garbage classification/Garbage classification/")
        print(f"       ├── cardboard/")
        print(f"       ├── glass/")
        print(f"       ├── metal/")
        print(f"       ├── paper/")
        print(f"       ├── plastic/")
        print(f"       └── trash/")
        return

    print(f"\n✅ 数据路径验证通过: {data_path}")

    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  计算设备: {device}")

    if device.type == 'cuda':
        print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"   显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
        print(f"   CUDA版本: {torch.version.cuda}")

    # 3. 加载数据
    print("\n📂 加载数据中...")
    train_loader, val_loader, classes = get_dataloaders(DATA_PATH, BATCH_SIZE)

    if len(train_loader.dataset) == 0:
        print("\n❌ 错误: 没有找到任何图片!")
        print("请检查类别文件夹内是否有图片文件")
        return

    # 4. 创建模型
    classifier = ResNeXtClassifier(len(classes), device)

    # 5. GPU性能测试（仅在CUDA可用时）
    if device.type == 'cuda':
        profile_gpu_performance(classifier, train_loader, device)

    # 6. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # 7. TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/classification_{timestamp}'
    writer = SummaryWriter(log_dir)

    # 记录超参数
    writer.add_text('Hyperparameters', f'''
    - 数据集路径: {DATA_PATH}
    - 批量大小: {BATCH_SIZE}
    - 学习率: {LR}
    - 训练轮数: {EPOCHS}
    - 混合精度: {USE_AMP}
    - 类别数: {len(classes)}
    - 类别: {classes}
    ''')

    # 8. 训练循环
    best_acc = 0
    print("\n" + "=" * 60)
    print("🚀 开始训练")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'=' * 50}")

        # 训练
        train_loss, train_acc = classifier.train_epoch(
            train_loader, criterion, optimizer, scaler, epoch, writer, USE_AMP
        )

        # 验证
        val_loss, val_acc = classifier.validate(val_loader, criterion, epoch, writer)

        # 打印结果
        print(f"\n📊 Epoch {epoch} 结果:")
        print(f"   训练 Loss: {train_loss:.4f} | 训练 Acc: {train_acc:.2f}%")
        print(f"   验证 Loss: {val_loss:.4f} | 验证 Acc: {val_acc:.2f}%")

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LR', current_lr, epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            classifier.save('best_classifier.pth')
            print(f"   ✨ 新的最佳模型! 准确率: {best_acc:.2f}%")

    # 9. 训练完成
    writer.close()

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"\n🏆 最佳验证准确率: {best_acc:.2f}%")
    print(f"📁 TensorBoard日志: {log_dir}")
    print(f"\n查看训练曲线:")
    print(f"   tensorboard --logdir runs")
    print(f"   然后打开浏览器访问: http://localhost:6006")

    # 10. 保存最终模型
    classifier.save('final_classifier.pth')

    # 11. 保存类别信息
    import json
    with open('classes.json', 'w') as f:
        json.dump({'classes': classes, 'class_to_idx': classifier.model.fc.class_to_idx if hasattr(classifier.model.fc,
                                                                                                   'class_to_idx') else {
            c: i for i, c in enumerate(classes)}}, f)
    print(f"📋 类别信息已保存: classes.json")


if __name__ == "__main__":
    main()