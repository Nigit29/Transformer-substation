from ultralytics import YOLO
import os
import shutil
import yaml
import torch

# ========== 1. 路径配置 ==========
# 修改为你的实际路径
dataset_path = r'E:\Pythonproject\trash_data'  # 主数据集路径

# 训练参数配置
model_name = 'yolov8n.pt'  # 可选: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
epochs = 20
imgsz = 640
batch_size = 16  # 根据你的GPU显存调整
workers = 2  # 数据加载线程数

# ========== 2. 创建数据集配置文件 ==========
# 检查目录结构
train_images_dir = os.path.join(dataset_path, 'train', 'images')
val_images_dir = os.path.join(dataset_path, 'val', 'images')

print("检查目录结构...")
print(f"训练图片目录: {train_images_dir} - 存在: {os.path.exists(train_images_dir)}")
print(f"验证图片目录: {val_images_dir} - 存在: {os.path.exists(val_images_dir)}")

if os.path.exists(train_images_dir):
    train_images_count = len([f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"训练图片数量: {train_images_count}")

if os.path.exists(val_images_dir):
    val_images_count = len([f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"验证图片数量: {val_images_count}")

# 创建 dataset.yaml 文件
yaml_path = os.path.join(dataset_path, 'data.yaml')

# YAML 内容（使用你的文件结构）
yaml_content = f"""
# 变电站检测数据集配置
path: {dataset_path}  # 数据集根目录
train: train/images   # 训练集路径（相对于path）
val: val/images       # 验证集路径（相对于path）
test: val/images      # 测试集路径（可选项）

# 类别数量
nc: 1

# 类别名称
names: ['foreign_object']

# 可选：下载命令/URL（如果有的话）
# download: https://ultralytics.com/assets/coco8.zip
"""

# 写入 YAML 文件
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"\n数据集配置文件已创建: {yaml_path}")


# ========== 3. 检查YOLO格式标签 ==========
def check_labels():
    print("\n检查标签文件...")
    for split in ['train', 'val']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(labels_dir):
            txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            print(f"{split}/labels: {len(txt_files)} 个标签文件")

            # 检查标签格式
            if txt_files:
                sample_file = os.path.join(labels_dir, txt_files[0])
                try:
                    with open(sample_file, 'r') as f:
                        first_line = f.readline().strip()
                        print(f"  示例标签格式: {first_line}")
                except:
                    print(f"  无法读取标签文件: {txt_files[0]}")
        else:
            print(f"警告: {labels_dir} 目录不存在")


check_labels()

# ========== 4. 训练模型 ==========
print("\n开始训练YOLOv8模型...")
print(f"使用模型: {model_name}")
print(f"训练轮数: {epochs}")
print(f"图片尺寸: {imgsz}")

try:
    # 加载预训练模型
    model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data=yaml_path,  # 数据集配置文件
        epochs=epochs,  # 训练轮数
        imgsz=imgsz,  # 输入图片尺寸
        batch=batch_size,  # 批大小
        workers=workers,  # 数据加载线程数
        name='变电站检测',  # 实验名称
        exist_ok=True,  # 允许覆盖现有实验
        save=True,  # 保存训练结果
        save_period=10,  # 每10个epoch保存一次检查点
        patience=50,  # 早停耐心值（50个epoch没有改进则停止）
        device='cuda' if torch.cuda.is_available() else 'cpu',  # 自动选择设备
        optimizer='auto',  # 自动选择优化器
        single_cls=True,  # 单类别模式（强制）
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率因子
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 预热轮数
        warmup_momentum=0.8,  # 预热动量
        box=7.5,  # 边界框损失权重
        cls=0.5,  # 分类损失权重
        dfl=1.5,  # DFL损失权重
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
        degrees=0.0,  # 旋转角度
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        shear=0.0,  # 剪切
        perspective=0.0,  # 透视变换
        flipud=0.0,  # 上下翻转概率
        fliplr=0.5,  # 左右翻转概率
        mosaic=1.0,  # mosaic数据增强概率
        mixup=0.0,  # mixup数据增强概率
        copy_paste=0.0  # 复制粘贴数据增强概率
    )

    print("训练完成！")

    # ========== 5. 保存最佳权重 ==========
    # 获取最新的训练实验目录
    import glob

    train_runs = sorted(glob.glob('runs/detect/变电站检测*'))

    if train_runs:
        latest_run = train_runs[-1]
        best_weights_path = os.path.join(latest_run, 'weights', 'best.pt')

        if os.path.exists(best_weights_path):
            # 复制到自定义位置
            saved_weights_path = os.path.join(dataset_path, 'best_model.pt')
            shutil.copy(best_weights_path, saved_weights_path)
            print(f"最佳模型已保存到: {saved_weights_path}")

            # 也保存最后一个权重
            last_weights_path = os.path.join(latest_run, 'weights', 'last.pt')
            if os.path.exists(last_weights_path):
                shutil.copy(last_weights_path, os.path.join(dataset_path, 'last_model.pt'))
        else:
            print("警告: 未找到最佳权重文件")
    else:
        print("警告: 未找到训练结果目录")

except Exception as e:
    print(f"训练过程中出现错误: {str(e)}")
    print("请检查:")
    print("1. 数据集路径是否正确")
    print("2. 图片和标签文件是否存在")
    print("3. YAML配置文件格式是否正确")
    print("4. 是否有足够的磁盘空间")


# ========== 6. 验证模型（可选） ==========
def validate_model(weights_path):
    if os.path.exists(weights_path):
        print(f"\n使用最佳模型进行验证...")
        model = YOLO(weights_path)
        metrics = model.val(data=yaml_path)
        print(f"验证完成！mAP50-95: {metrics.box.map:.4f}")
    else:
        print(f"权重文件不存在: {weights_path}")

# 如果需要验证，取消下面的注释
# validate_model(os.path.join(dataset_path, 'best_model.pt'))