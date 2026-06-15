# syjdeeplearning
深度学习期末综合实践项目 

项目简介
本项目为深度学习技术与应用课程期末综合实践大作业，选用方向一：融合 ResNeXt 与 DeepLabV3 的智能视觉分析系统，基于 PyTorch 框架实现图像分类 + 语义分割双任务视觉系统。硬件环境为本地 NVIDIA RTX 4060（8GB），软件基于 Python 3.10、PyTorch 2.x、CUDA 12.x 开发，以垃圾分类数据集完成模型训练、调优、性能测试与可视化分析，同时完成混合精度加速、超参数消融实验、TensorBoard 训练监控等工程化实践。
学生信息
姓名：宋盈建
学号：2330200164
班级：23 级人工智能工程技术 11 班
一、环境配置
1. 软硬件环境
表格
类别	配置信息
操作系统	Windows 10/11
硬件	NVIDIA RTX 4060 8GB
Python	3.10
框架	PyTorch 2.x、TorchVision
加速环境	CUDA 12.x
可视化工具	TensorBoard、Matplotlib
版本管理	Git
2. 依赖安装
方式 1：Conda 环境（推荐）
bash
运行
# 创建并激活环境
conda create -n dl_final python=3.10
conda activate dl_final

# 安装深度学习依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib pillow pathlib numpy tensorboard
conda install git
方式 2：批量安装（requirements.txt）
在项目根目录新建 requirements.txt，写入以下内容：
txt
torch>=2.0
torchvision
numpy
matplotlib
pillow
tensorboard
pathlib
执行安装：
bash
运行
pip install -r requirements.txt
3. 环境校验
运行以下命令校验 GPU 可用性：
python
运行
import torch
print(torch.cuda.is_available())   # 输出 True 代表GPU可用
print(torch.cuda.get_device_name(0)) # 输出 NVIDIA GeForce RTX 4060
二、项目目录结构
plaintext
dl_final_project/
├── data/                    # 数据集根目录
│   ├── Garbage classification/  # 垃圾分类数据集
│   └── VOCdevkit/            # 语义分割数据集子集
├── runs/                    # TensorBoard 日志文件（训练日志、曲线）
├── weights/                 # 训练保存的模型权重
│   ├── best_classifier.pth
│   ├── best_classifier_exp1.pth
│   ├── final_classifier.pth
│   └── 分割模型权重文件
├── classes.json             # 垃圾分类类别映射文件
├── train_classification.py  # ResNeXt 分类训练主脚本
├── train_segmentation.py    # DeepLabV3 语义分割训练脚本
├── test_classification.py   # 分类模型交互式测试脚本
├── eval_model.py            # 超参数实验 & 模型评估脚本
├── README.md                # 项目说明文档
└── requirements.txt         # 环境依赖清单
三、数据集说明
1. 数据集来源与划分
主任务：垃圾分类数据集，共 6 个类别：cardboard、glass、metal、paper、plastic、trash
划分比例：训练集 / 验证集 / 测试集
测试集总量：5046 张图像
配套文件：one-indexed-files.txt、zero-indexed-files.txt 等索引文件，用于数据集加载
2. 数据预处理流水线
数据清洗：过滤损坏、无效图片
数据增强：RandomResizedCrop、RandomHorizontalFlip、ColorJitter
归一化：使用 ImageNet 标准均值与方差
封装：基于 torch.utils.data.DataLoader 批量加载数据
四、核心模型介绍
1. 图像分类模块（ResNeXt）
主干网络：ResNeXt 预训练模型，采用迁移学习 + 微调策略
训练策略：冻结主干部分参数，仅微调分类头；后期全局微调
正则化：多层 Dropout + AdamW 权重衰减（L2 正则）+ 早停策略
2. 语义分割模块（DeepLabV3）
核心结构：DeepLabV3 + ASPP 空洞空间金字塔池化模块
特征共享：与 ResNeXt 共享主干特征提取网络，实现多任务学习
评价指标：分割任务使用 mIoU 作为核心评估指标
3. 混合精度训练
启用 torch.cuda.amp 混合精度（AMP），对比 FP32 全精度训练：
表格
精度模式	单 Batch 耗时	显存占用	加速比
FP32	74.47ms	0.44GB	1.00x
AMP	36.94ms	0.29GB	2.02x
结论：AMP 大幅降低显存占用、提升训练速度，加速比约 2.02 倍。
五、超参数消融实验（共 3 组）
固定随机种子保证实验可复现，针对学习率、BatchSize、Dropout开展对比实验：
表格
实验编号	学习率	BatchSize	Dropout 配置	优化器	测试集准确率
实验 1（基线）	0.001	32	0.5 / 0.3	AdamW	98.71%
实验 2（小学习率）	0.0001	32	0.5 / 0.3	AdamW	97.42%
实验 3（最终模型）	0.001	16	0.3 / 0.2	AdamW	99.11%
实验结论
学习率过小会导致模型收敛变慢、最终精度下降；
适当减小 BatchSize、降低 Dropout 系数，在本数据集上精度最优；
三组实验训练集与验证集误差差距较小，无过拟合 / 欠拟合问题。
正则化策略
Dropout：随机丢弃神经元，减弱过拟合；
L2 正则：AdamW 设置 weight_decay=1e-4，约束权重大小；
早停（Early Stop）：保存验证集最优模型，避免过度训练；
数据增强：扩充数据多样性，提升模型泛化能力。
六、运行指南
1. 启动训练
（1）训练垃圾分类分类模型
bash
运行
python train_classification.py
训练日志自动保存至 runs/，模型权重保存至 weights/
自动启用 AMP 混合精度训练、TensorBoard 日志记录
（2）训练语义分割模型
bash
运行
python train_segmentation.py
2. 启动 TensorBoard 可视化
bash
运行
tensorboard --logdir=./runs --port=6006
浏览器访问：http://localhost:6006
可查看：Loss 曲线、准确率曲线、学习率变化、特征图、热力图等。
3. 模型评估（超参数实验批量测试）
bash
运行
python eval_model.py
自动加载多组权重，输出各类别准确率、汇总实验报表。
4. 交互式测试（单张 / 批量图片推理）
bash
运行
python test_classification.py
功能菜单：
测试单张图片（可视化预测结果 + 概率分布）
批量测试文件夹内所有图片
退出程序
七、功能演示说明
单图推理：输入图片路径，输出预测类别、置信度，同时绘制原图 + 类别概率柱状图；
批量推理：遍历指定文件夹下 jpg/png 图片，批量输出预测结果；
可视化监控：TensorBoard 全程监控训练 Loss、Accuracy、学习率变化；
性能监控：代码内置计时、显存统计，对比 CPU / RTX 4060 GPU 推理 FPS。
八、项目问题与优化思路
1. 项目现存问题
部分小众类别样本数量偏少，单类别识别准确率存在波动；
语义分割分支在复杂背景下分割边缘不够精细；
推理代码未封装为 Web 接口，暂未完成 Flask/FastAPI 部署。
2. 优化改进思路
对少样本类别做数据增强 + 重采样，平衡数据集分布；
调优 ASPP 模块空洞卷积参数，增加边缘特征提取能力；
基于 Flask/FastAPI 封装推理接口，实现网页端在线预测；
引入 Grad-CAM 类激活图，可视化模型关注区域。
九、项目总结
本项目完成 ResNeXt 图像分类与 DeepLabV3 语义分割双任务融合系统，完整实现数据工程、模型搭建、迁移学习、混合精度训练、超参数调优、可视化监控、性能分析全流程深度学习工程实践。
依托 RTX 4060 硬件完成 GPU 性能剖析，验证了 AMP 混合精度对训练加速与显存优化的效果；通过多组消融实验验证了学习率、BatchSize、正则化策略对模型精度的影响。项目代码模块化、注释完整、可一键复现，满足课程综合实践全部要求。
