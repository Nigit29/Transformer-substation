"""
批量检测 + 统计准确率 - 用于论文
"""
import cv2
import os
from pathlib import Path
from detector import Detector
import csv

# 配置
model_path = 'biandianzhan/weights/best.pt'
dataset_path = r'E:\BaiduNetdiskDownload\trash_data'
output_dir = 'paper_results'
conf_threshold = 0.60
iou_threshold = 0.45

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

print("=" * 50)
print("批量检测脚本 - 用于论文")
print("=" * 50)
print(f"置信度阈值: {conf_threshold}")
print(f"面积过滤: < 30% 图片面积")
print()

# 初始化检测器
print("加载模型...")
detector = Detector(model_path)

# 获取验证集图片
test_images = list(Path(dataset_path).glob('val/images/*.jpg'))
if not test_images:
    test_images = list(Path(dataset_path).glob('train/images/*.jpg'))

print(f"找到 {len(test_images)} 张图片\n")

# 统计数据
results_list = []
saved_count = 0

print("开始检测...")
for idx, img_path in enumerate(test_images, 1):
    img_name = img_path.name

    # 读取图片
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    # 检测
    result = detector.detect_image(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    # 检查标签
    label_path = Path(dataset_path) / 'val' / 'labels' / (img_path.stem + '.txt')
    has_ground_truth = False
    gt_count = 0

    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if lines:
                has_ground_truth = True
                gt_count = len(lines)

    # 统计
    predicted_count = result['count'] if result else 0
    is_tp = has_ground_truth and predicted_count > 0  # 真阳性
    is_fp = not has_ground_truth and predicted_count > 0  # 假阳性
    is_fn = has_ground_truth and predicted_count == 0  # 假阴性
    is_tn = not has_ground_truth and predicted_count == 0  # 真阴性

    results_list.append({
        '图片名': img_name,
        '标签数': gt_count,
        '检测数': predicted_count,
        '最高置信度': result['max_conf'] if result else 0,
        '真阳性(TP)': is_tp,
        '假阳性(FP)': is_fp,
        '假阴性(FN)': is_fn,
        '真阴性(TN)': is_tn
    })

    # 保存检测结果图
    if result and predicted_count > 0:
        output_path = f"{output_dir}/result_{saved_count+1:02d}.jpg"
        cv2.imwrite(output_path, result['image'])

        # 保存标注信息
        info_path = f"{output_dir}/result_{saved_count+1:02d}.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"图片: {img_name}\n")
            f.write(f"标签数: {gt_count}\n")
            f.write(f"检测数: {predicted_count}\n")
            f.write(f"最高置信度: {result['max_conf']:.3f}\n")
            f.write(f"平均置信度: {result['avg_conf']:.3f}\n")
            f.write(f"类型: {'真阳性(TP)' if is_tp else '假阳性(FP)'}\n")

        saved_count += 1
        print(f"[{idx}] {img_name} - 标签:{gt_count} 检测:{predicted_count} 置信度:{result['max_conf']:.3f}")

    if saved_count >= 15:  # 保存15张
        break

print(f"\n检测完成！保存了 {saved_count} 张结果图片到 {output_dir}/")

# 计算统计指标
tp = sum(1 for r in results_list if r['真阳性(TP)'])
fp = sum(1 for r in results_list if r['假阳性(FP)'])
fn = sum(1 for r in results_list if r['假阴性(FN)'])
tn = sum(1 for r in results_list if r['真阴性(TN)'])

print("\n" + "=" * 50)
print("检测结果统计")
print("=" * 50)
print(f"总图片数: {len(results_list)}")
print(f"真阳性(TP): {tp} - 有异物且检测到")
print(f"假阳性(FP): {fp} - 无异物但误检")
print(f"假阴性(FN): {fn} - 有异物但漏检")
print(f"真阴性(TN): {tn} - 无异物且未检测")
print()

# 计算指标
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("性能指标:")
print(f"精确率(Precision): {precision:.3f}")
print(f"召回率(Recall): {recall:.3f}")
print(f"F1分数: {f1:.3f}")

# 保存统计结果到CSV
csv_path = f'{output_dir}/statistics.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['图片名', '标签数', '检测数', '最高置信度', '真阳性(TP)', '假阳性(FP)', '假阴性(FN)', '真阴性(TN)'])
    for r in results_list:
        writer.writerow([
            r['图片名'], r['标签数'], r['检测数'],
            f"{r['最高置信度']:.3f}",
            r['真阳性(TP)'], r['假阳性(FP)'], r['假阴性(FN)'], r['真阴性(TN)']
        ])

print(f"\n统计数据已保存到 {csv_path}")
