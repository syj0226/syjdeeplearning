"""
垃圾分类模型测试脚本
测试训练好的 ResNeXt 模型
运行方式: python test_classification.py
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 然后是其他导入
import torch
import json
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import json
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

# ============ 配置 ============
MODEL_PATH = "best_classifier.pth"  # 使用最佳模型
CLASSES_PATH = "classes.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============ 加载模型 ============
def load_model(num_classes=6):
    """加载训练好的模型"""
    # 动态导入模型类
    import importlib.util

    # 加载训练脚本中的类
    spec = importlib.util.spec_from_file_location("train_classification", "train_classification.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    classifier = train_module.ResNeXtClassifier(num_classes, DEVICE)
    classifier.load(MODEL_PATH)
    classifier.model.eval()

    return classifier

# ============ 图片预处理 ============
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# ============ 单张图片预测 ============
def predict_single_image(image_path, classifier, classes):
    """预测单张图片"""

    # 加载图片
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        output = classifier.model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    # 打印结果
    print(f"\n📷 图片: {Path(image_path).name}")
    print(f"✅ 预测类别: {classes[pred_idx]}")
    print(f"📊 置信度: {confidence:.2%}")

    # 打印所有类别概率
    print("\n所有类别概率:")
    for i, cls in enumerate(classes):
        print(f"   {cls:10s}: {probs[0][i].item():.2%}")

    return classes[pred_idx], confidence, probs

# ============ 显示图片和预测结果 ============
def show_prediction(image_path, classifier, classes):
    """显示图片和预测结果"""

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = classifier.model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 使用英文标题（这就是方法二的修改）
    ax1.imshow(np.array(img))
    ax1.set_title(f"Prediction: {classes[pred_idx]}\nConfidence: {confidence:.2%}")
    ax1.axis('off')

    probs_np = probs[0].cpu().numpy()
    colors = ['green' if i == pred_idx else 'gray' for i in range(len(classes))]
    ax2.bar(classes, probs_np, color=colors)
    ax2.set_ylabel('Probability')
    ax2.set_title('Class Probability Distribution')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# ============ 批量测试 ============
def batch_test(test_dir, classifier, classes):
    """测试整个文件夹"""
    test_dir = Path(test_dir)

    if not test_dir.exists():
        print(f"❌ 目录不存在: {test_dir}")
        return

    results = []
    for img_path in list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")):
        pred, conf, _ = predict_single_image(str(img_path), classifier, classes)
        results.append({
            'file': img_path.name,
            'prediction': pred,
            'confidence': conf
        })

    print(f"\n📊 批量测试结果 ({len(results)} 张图片)")
    print("-" * 50)
    for r in results:
        print(f"  {r['file']:30s} -> {r['prediction']:10s} (置信度: {r['confidence']:.2%})")

    return results

# ============ 交互式菜单 ============
def print_menu():
    print("\n" + "-"*40)
    print("垃圾分类模型测试")
    print("-"*40)
    print("1. 测试单张图片")
    print("2. 测试整个文件夹")
    print("3. 退出")
    print("-"*40)

# ============ 主函数 ============
def main():
    print("="*50)
    print("🧪 垃圾分类模型测试")
    print("="*50)

    # 检查模型文件
    if not Path(MODEL_PATH).exists():
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请先运行 train_classification.py 训练模型")
        return

    # 检查类别文件
    if not Path(CLASSES_PATH).exists():
        print(f"❌ 类别文件不存在: {CLASSES_PATH}")
        return

    # 加载类别
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        classes_data = json.load(f)
        classes = classes_data['classes']

    print(f"\n📋 类别: {classes}")

    # 加载模型
    try:
        classifier = load_model(len(classes))
        print(f"✅ 模型加载成功: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 交互式测试
    while True:
        print_menu()
        choice = input("请选择 (1/2/3): ").strip()

        if choice == '1':
            img_path = input("请输入图片路径: ").strip()
            if Path(img_path).exists():
                show_prediction(img_path, classifier, classes)
            else:
                print(f"❌ 文件不存在: {img_path}")

        elif choice == '2':
            dir_path = input("请输入文件夹路径: ").strip()
            batch_test(dir_path, classifier, classes)

        elif choice == '3':
            print("👋 再见!")
            break

        else:
            print("❌ 无效输入，请输入 1、2 或 3")

if __name__ == "__main__":
    main()