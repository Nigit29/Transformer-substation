"""
YOLOv8-SE 检测器
使用带SE注意力机制的YOLOv8模型进行检测
"""

import cv2
import numpy as np
from ultralytics import YOLO


class DetectorSE:
    """带SE注意力机制的YOLOv8检测器"""

    def __init__(self, model_path='yolov8n_se_best.pt'):
        """
        初始化SE-YOLO检测器

        Args:
            model_path: SE-YOLO模型权重文件路径
        """
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """加载SE-YOLO模型"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ SE-YOLO模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"✗ SE-YOLO模型加载失败: {e}")
            return False

    def detect_image(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        检测单张图片

        Args:
            image: 输入图片 (numpy数组或文件路径)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            dict: 检测结果
        """
        if self.model is None:
            return None

        # 如果是文件路径，读取图片
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                print(f"无法读取图片: {image}")
                return None

        # 获取图片尺寸
        img_height, img_width = image.shape[:2]
        img_area = img_height * img_width

        # 执行检测
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # 处理结果
        detections = []
        max_conf = 0.0
        confidences = []

        annotated_image = image.copy()

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # 获取边界框
                x1, y1, x2, y2 = box.xyxy[0].astype(int)

                # 获取类别和置信度
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # 过滤大框（背景误检）
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / img_area
                if area_ratio > 0.3:
                    continue

                # 获取类别名称
                label = self.model.names[cls] if cls in self.model.names else f'class_{cls}'

                # 更新统计信息
                max_conf = max(max_conf, conf)
                confidences.append(conf)

                # 保存检测信息
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'label': label,
                    'confidence': conf,
                    'class_id': cls
                })

                # 绘制边界框（使用蓝色表示SE模型）
                color = (255, 0, 0)  # 蓝色
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label_text = f'{label}: {conf:.2f}'
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image,
                             (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1),
                             color, -1)
                cv2.putText(annotated_image, label_text,
                           (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 计算平均置信度
        avg_conf = np.mean(confidences) if confidences else 0.0

        return {
            'image': annotated_image,
            'results': detections,
            'count': len(detections),
            'max_conf': max_conf,
            'avg_conf': round(avg_conf, 4)
        }

    def compare_models(self, image, normal_model_path='biandianzhan/weights/best.pt'):
        """
        对比普通YOLOv8和SE-YOLOv8的检测结果

        Args:
            image: 输入图片
            normal_model_path: 普通YOLOv8模型路径

        Returns:
            dict: 对比结果
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return None

        # 普通YOLOv8检测
        normal_detector = Detector(normal_model_path) if os.path.exists(normal_model_path) else None
        normal_result = normal_detector.detect_image(image) if normal_detector else None

        # SE-YOLOv8检测
        se_result = self.detect_image(image)

        # 创建对比图
        if normal_result and se_result:
            # 左右拼接对比图
            h, w = image.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = normal_result['image']
            comparison[:, w:] = se_result['image']

            # 添加标题
            cv2.putText(comparison, "YOLOv8", (w // 2 - 50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "YOLOv8-SE", (w * 3 // 2 - 80, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            return {
                'comparison_image': comparison,
                'normal_count': normal_result['count'],
                'se_count': se_result['count'],
                'normal_avg_conf': normal_result['avg_conf'],
                'se_avg_conf': se_result['avg_conf']
            }

        return None


def test_se_model():
    """测试SE模型"""
    print("========== YOLOv8-SE 模型测试 ==========")

    # 创建检测器
    detector = DetectorSE('yolov8n_se_best.pt')

    if not detector.model:
        print("请先训练模型！")
        return

    # 测试图片
    test_image = 'detection_result.jpg'

    if os.path.exists(test_image):
        result = detector.detect_image(test_image)

        if result:
            print(f"\n检测结果:")
            print(f"检测到物体数量: {result['count']}")
            print(f"最高置信度: {result['max_conf']:.4f}")
            print(f"平均置信度: {result['avg_conf']:.4f}")

            # 保存结果
            cv2.imwrite('se_detection_result.jpg', result['image'])
            print(f"结果已保存到: se_detection_result.jpg")

            # 对比测试
            print("\n========== 模型对比 ==========")
            comparison = detector.compare_models(test_image)

            if comparison:
                print(f"\n普通YOLOv8: 检测{comparison['normal_count']}个目标, 平均置信度{comparison['normal_avg_conf']:.4f}")
                print(f"YOLOv8-SE: 检测{comparison['se_count']}个目标, 平均置信度{comparison['se_avg_conf']:.4f}")

                cv2.imwrite('comparison_result.jpg', comparison['comparison_image'])
                print(f"对比图已保存到: comparison_result.jpg")
    else:
        print(f"测试图片不存在: {test_image}")


if __name__ == '__main__':
    import os
    test_se_model()
