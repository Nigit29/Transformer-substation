"""
检测引擎模块 - 封装YOLO检测逻辑
"""
import cv2
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='biandianzhan/weights/best.pt'):
        """
        初始化检测器

        Args:
            model_path: YOLO模型权重文件路径
        """
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_path)
            print(f"模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def detect_image(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        检测单张图片

        Args:
            image: 输入图片 (numpy数组或文件路径)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            dict: {
                'image': 绘制了检测框的图片,
                'results': 检测结果列表,
                'count': 检测到的物体数量,
                'max_conf': 最高置信度,
                'avg_conf': 平均置信度
            }
        """
        if self.model is None:
            return None

        # 如果是文件路径，读取图片
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                print(f"无法读取图片: {image}")
                return None

        # 获取图片尺寸用于面积过滤
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

                # ===== 快速修复：过滤掉大框（背景误检）=====
                # 计算检测框面积
                box_area = (x2 - x1) * (y2 - y1)
                # 计算面积占比
                area_ratio = box_area / img_area
                # 如果检测框超过图片面积的 30%，认为是误检（背景），跳过
                if area_ratio > 0.3:
                    continue
                # ====================================================

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

                # 绘制边界框
                color = (0, 255, 0)  # 绿色
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
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 计算平均置信度
        avg_conf = np.mean(confidences) if confidences else 0.0

        return {
            'image': annotated_image,
            'results': detections,
            'count': len(detections),
            'max_conf': max_conf,
            'avg_conf': round(avg_conf, 4)
        }

    def detect_folder(self, folder_path, conf_threshold=0.25, iou_threshold=0.45,
                     progress_callback=None):
        """
        检测文件夹中的所有图片

        Args:
            folder_path: 文件夹路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            progress_callback: 进度回调函数 (current, total)

        Returns:
            list: 检测结果列表
        """
        import os
        from pathlib import Path

        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"文件夹中未找到图片: {folder_path}")
            return []

        results = []
        total = len(image_files)

        for idx, img_file in enumerate(image_files):
            img_path = str(img_file)
            result = self.detect_image(img_path, conf_threshold, iou_threshold)

            if result:
                result['source'] = img_path
                results.append(result)

            # 调用进度回调
            if progress_callback:
                progress_callback(idx + 1, total)

        return results

    def detect_video_frame(self, frame, conf_threshold=0.25, iou_threshold=0.45):
        """
        检测视频帧

        Args:
            frame: 视频帧
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            dict: 检测结果（同 detect_image）
        """
        return self.detect_image(frame, conf_threshold, iou_threshold)

    def detect_video_file(self, video_path, conf_threshold=0.25, iou_threshold=0.45,
                         output_path=None, progress_callback=None):
        """
        检测视频文件

        Args:
            video_path: 视频文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            output_path: 输出视频路径（可选）
            progress_callback: 进度回调函数 (current, total, frame)

        Returns:
            dict: {
                'total_frames': 总帧数,
                'processed_frames': 处理帧数,
                'total_objects': 检测到的总物体数,
                'avg_confidence': 平均置信度
            }
        """
        # 打开视频（尝试多个后端）
        cap = None
        backend_options = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF] if cv2.__version__.startswith('4') else [cv2.CAP_ANY]

        for backend in backend_options:
            cap = cv2.VideoCapture(video_path, backend)
            if cap is not None and cap.isOpened():
                # 测试能否读取帧
                ret, test_frame = cap.read()
                if ret:
                    # 重置视频位置到开始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                else:
                    cap.release()
                    cap = None
            elif cap is not None:
                cap.release()
                cap = None

        if cap is None or not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return None

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

        # 创建输出视频 writer
        video_writer = None
        if output_path:
            # 尝试多个编码器，优先使用兼容性好的
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'mp4v'),  # MP4标准编码器
                cv2.VideoWriter_fourcc(*'XVID'),   # XVID编码器
                cv2.VideoWriter_fourcc(*'MJPG'),   # MJPEG编码器
                cv2.VideoWriter_fourcc(*'avc1'),   # H.264编码器
            ]

            video_writer = None
            for fourcc in fourcc_options:
                try:
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if video_writer.isOpened():
                        print(f"视频编码器: {fourcc}")
                        break
                except:
                    video_writer = None

            if video_writer is None or not video_writer.isOpened():
                print(f"警告: 无法创建视频写入器，尝试使用默认编码器")
                # 最后尝试默认编码器
                video_writer = cv2.VideoWriter(output_path, 0, fps, (width, height))

        # 统计信息
        total_objects = 0
        all_confidences = []
        processed_frames = 0

        # 逐帧检测
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测当前帧
            result = self.detect_video_frame(frame, conf_threshold, iou_threshold)

            if result:
                total_objects += result['count']
                all_confidences.extend([d['confidence'] for d in result['results']])
                processed_frames += 1

                # 写入输出视频
                if video_writer:
                    video_writer.write(result['image'])

                # 调用进度回调
                if progress_callback:
                    progress_callback(frame_count + 1, total_frames, result['image'])

            frame_count += 1

        # 清理
        cap.release()
        if video_writer:
            video_writer.release()

        # 计算统计信息
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0

        return {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'total_objects': total_objects,
            'avg_confidence': round(avg_confidence, 4),
            'max_confidence': max(all_confidences) if all_confidences else 0.0
        }

    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return None

        return {
            'path': self.model_path,
            'classes': self.model.names,
            'num_classes': len(self.model.names)
        }
