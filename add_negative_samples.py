"""
为没有标签的图片创建空标签文件，添加负样本
"""
import os

# 数据集路径
dataset_path = r'E:\BaiduNetdiskDownload\trash_data'

print("=== 添加负样本（空标签文件）===\n")

for split in ['train', 'val']:
    labels_dir = os.path.join(dataset_path, split, 'labels')
    images_dir = os.path.join(dataset_path, split, 'images')

    print(f"处理 {split.upper()} 集...")

    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # 获取所有标签文件
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

    # 找出没有标签的图片
    missing_labels = []
    for img_file in image_files:
        img_base = os.path.splitext(img_file)[0]
        if img_base not in label_files:
            missing_labels.append(img_file)

    # 为这些图片创建空标签文件
    created_count = 0
    for img_file in missing_labels:
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
        # 创建空标签文件
        with open(label_path, 'w') as f:
            pass  # 空文件表示无异物
        created_count += 1

    print(f"  为 {created_count} 张图片创建了空标签文件")
    print(f"  总图片数: {len(image_files)}")
    print(f"  原有标签: {len(label_files)}")
    print(f"  新增负样本: {created_count}")
    print(f"  现在标签总数: {len(label_files) + created_count}")
    print()

print("=== 完成 ===")
print("\n负样本添加成功！")
print("请重新训练模型以获得正确的结果。")
