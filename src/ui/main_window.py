"""
主检测窗口 - 变电站异物检测主界面
"""
import sys
import os
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QTabWidget, QStatusBar, QGroupBox, QSplitter,
    QHeaderView, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

from detector import Detector
from statistics import Statistics
from database import Database
from config import Config


class MainWindow(QMainWindow):
    def __init__(self, user_info):
        """
        初始化主窗口

        Args:
            user_info: 当前用户信息字典
        """
        super().__init__()

        self.user_info = user_info
        self.detector = Detector(Config.MODEL_PATH)
        self.statistics = Statistics(Config.DB_PATH)
        self.database = Database(Config.DB_PATH)

        # 检测状态
        self.current_image = None
        self.current_image_path = None
        self.camera = None
        self.camera_timer = None
        self.video_file = None
        self.is_detecting = False

        # 初始化界面
        self.init_ui()
        self.update_statistics()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("变电站异物检测系统")
        self.setGeometry(100, 100, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 顶部栏 - 用户信息和退出按钮
        top_bar = QHBoxLayout()
        user_label = QLabel(f"用户: {self.user_info['username']} ({self.user_info['role']})")
        user_label.setFont(QFont("Arial", 12, QFont.Bold))
        logout_button = QPushButton("退出")
        logout_button.clicked.connect(self.logout)

        top_bar.addWidget(user_label)
        top_bar.addStretch()
        top_bar.addWidget(logout_button)
        main_layout.addLayout(top_bar)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧 - 显示区域
        left_panel = self.create_display_panel()
        splitter.addWidget(left_panel)

        # 右侧 - 控制和统计面板
        right_panel = self.create_control_panel()
        splitter.addWidget(right_panel)

        # 设置分割器比例
        splitter.setSizes([700, 400])

        main_layout.addWidget(splitter)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def create_display_panel(self):
        """创建显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setText("请加载图片或打开摄像头")
        self.image_label.setFont(QFont("Arial", 14))
        layout.addWidget(self.image_label)

        # 进度条（用于视频/文件夹检测）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 检测信息标签
        self.detection_info_label = QLabel("检测信息: 等待检测")
        self.detection_info_label.setStyleSheet("padding: 5px; background-color: #e0e0e0; border-radius: 5px;")
        layout.addWidget(self.detection_info_label)

        return panel

    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 创建选项卡
        tab_widget = QTabWidget()

        # 检测控制选项卡
        detect_tab = self.create_detection_tab()
        tab_widget.addTab(detect_tab, "检测控制")

        # 统计记录选项卡
        stats_tab = self.create_statistics_tab()
        tab_widget.addTab(stats_tab, "统计记录")

        layout.addWidget(tab_widget)

        return panel

    def create_detection_tab(self):
        """创建检测控制选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 输入方式选择
        input_group = QGroupBox("输入方式")
        input_layout = QVBoxLayout()

        # 图片按钮
        image_layout = QHBoxLayout()
        self.load_image_button = QPushButton("加载图片")
        self.load_image_button.clicked.connect(self.load_image)
        image_layout.addWidget(self.load_image_button)
        input_layout.addLayout(image_layout)

        # 文件夹按钮
        folder_layout = QHBoxLayout()
        self.load_folder_button = QPushButton("加载文件夹")
        self.load_folder_button.clicked.connect(self.load_folder)
        folder_layout.addWidget(self.load_folder_button)
        input_layout.addLayout(folder_layout)

        # 摄像头按钮
        camera_layout = QHBoxLayout()
        self.open_camera_button = QPushButton("打开摄像头")
        self.open_camera_button.clicked.connect(self.toggle_camera)
        camera_layout.addWidget(self.open_camera_button)
        input_layout.addLayout(camera_layout)

        # 视频按钮
        video_layout = QHBoxLayout()
        self.load_video_button = QPushButton("加载视频")
        self.load_video_button.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_video_button)
        input_layout.addLayout(video_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 检测控制
        control_group = QGroupBox("检测控制")
        control_layout = QVBoxLayout()

        self.start_detect_button = QPushButton("开始检测")
        self.start_detect_button.clicked.connect(self.start_detection)
        self.start_detect_button.setEnabled(False)
        control_layout.addWidget(self.start_detect_button)

        self.stop_detect_button = QPushButton("停止")
        self.stop_detect_button.clicked.connect(self.stop_detection)
        self.stop_detect_button.setEnabled(False)
        control_layout.addWidget(self.stop_detect_button)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 实时统计
        stats_group = QGroupBox("实时统计")
        stats_layout = QVBoxLayout()
        self.realtime_stats_label = QLabel("检测次数: 0\n异物总数: 0\n平均置信度: 0.00")
        self.realtime_stats_label.setFont(QFont("Arial", 10))
        stats_layout.addWidget(self.realtime_stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()

        return tab

    def create_statistics_tab(self):
        """创建统计记录选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 总体统计
        overall_group = QGroupBox("总体统计")
        overall_layout = QVBoxLayout()
        self.overall_stats_label = QLabel("加载中...")
        overall_layout.addWidget(self.overall_stats_label)
        overall_group.setLayout(overall_layout)
        layout.addWidget(overall_group)

        # 历史记录表格
        records_group = QGroupBox("检测记录")
        records_layout = QVBoxLayout()

        self.records_table = QTableWidget()
        self.records_table.setColumnCount(5)
        self.records_table.setHorizontalHeaderLabels(["时间", "类型", "来源", "异物数", "置信度"])
        self.records_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.records_table.setAlternatingRowColors(True)
        records_layout.addWidget(self.records_table)

        records_group.setLayout(records_layout)
        layout.addWidget(records_group)

        return tab

    # ========== 图片操作 ==========

    def load_image(self):
        """加载单张图片"""
        self.stop_camera()
        self.stop_video()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)"
        )

        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
            self.start_detect_button.setEnabled(True)
            self.status_bar.showMessage(f"已加载: {os.path.basename(file_path)}")

    def display_image(self, image):
        """在界面上显示图片"""
        if image is None:
            return

        try:
            # 转换颜色空间
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            # 创建QImage - 使用copy()确保数据不被过早释放
            qt_image = QImage(rgb_image.copy().data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            if pixmap.isNull():
                print("QPixmap为空")
                return

            # 获取标签的可用大小
            label_size = self.image_label.size()
            if label_size.width() <= 0 or label_size.height() <= 0:
                # 如果标签大小无效，使用默认大小
                scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                # 缩放以适应显示区域
                scaled_pixmap = pixmap.scaled(
                    label_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"显示图片失败: {e}")

    # ========== 文件夹操作 ==========

    def load_folder(self):
        """加载文件夹进行批量检测"""
        self.stop_camera()
        self.stop_video()

        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")

        if folder_path:
            self.status_bar.showMessage("正在处理文件夹...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # 执行检测
            def progress_callback(current, total):
                self.progress_bar.setValue(int(current / total * 100))
                self.status_bar.showMessage(f"正在处理: {current}/{total}")

            try:
                results = self.detector.detect_folder(
                    folder_path,
                    Config.CONF_THRESHOLD,
                    Config.IOU_THRESHOLD,
                    progress_callback
                )

                # 处理结果
                if results:
                    # 显示最后一张结果
                    last_result = results[-1]
                    self.display_image(last_result['image'])
                    self.save_detection_record('folder', folder_path, last_result)
                    self.update_statistics()

                    self.status_bar.showMessage(f"文件夹处理完成，共 {len(results)} 张图片")
                else:
                    self.status_bar.showMessage("文件夹中没有可处理的图片")
            except Exception as e:
                self.status_bar.showMessage(f"处理失败: {str(e)}")
                print(f"文件夹检测错误: {e}")

            self.progress_bar.setVisible(False)

    # ========== 摄像头操作 ==========

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.camera is None or not self.camera.isOpened():
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """启动摄像头"""
        try:
            self.status_bar.showMessage("正在打开摄像头...")

            # 自动检测可用摄像头
            camera_opened = False
            camera_backend = None

            # Windows 优先使用 DirectShow
            if sys.platform == 'win32':
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
            else:
                backends = [cv2.CAP_ANY]

            # 尝试打开摄像头
            for backend in backends:
                for cam_idx in range(5):  # 尝试前5个摄像头索引
                    self.camera = cv2.VideoCapture(cam_idx, backend)
                    if self.camera.isOpened():
                        # 测试是否能读取帧
                        ret, frame = self.camera.read()
                        if ret:
                            camera_opened = True
                            camera_backend = backend
                            self.status_bar.showMessage(f"摄像头已启动 (索引: {cam_idx})")
                            break
                    self.camera.release()

                if camera_opened:
                    break

            if not camera_opened:
                QMessageBox.warning(self, "错误",
                    "无法打开摄像头\n\n请检查：\n1. 摄像头是否连接\n2. 摄像头是否被其他程序占用\n3. Windows摄像头隐私设置\n4. 摄像头驱动是否正常")
                return

            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)

            self.open_camera_button.setText("关闭摄像头")
            self.start_detect_button.setEnabled(self.is_detecting)

            # 创建定时器
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self.process_camera_frame)
            self.camera_timer.start(int(1000 / Config.CAMERA_FPS))

        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动摄像头失败: {str(e)}\n\n{type(e).__name__}")

    def stop_camera(self):
        """关闭摄像头"""
        if self.camera_timer:
            self.camera_timer.stop()
            self.camera_timer = None

        if self.camera:
            self.camera.release()
            self.camera = None

        self.open_camera_button.setText("打开摄像头")
        self.status_bar.showMessage("摄像头已关闭")

    def stop_video(self):
        """停止视频处理"""
        self.current_video_path = None
        self.status_bar.showMessage("视频已停止")

    def process_camera_frame(self):
        """处理摄像头帧"""
        ret, frame = self.camera.read()
        if not ret:
            return

        if self.is_detecting:
            result = self.detector.detect_video_frame(frame, Config.CONF_THRESHOLD, Config.IOU_THRESHOLD)
            if result:
                self.display_image(result['image'])
                self.update_detection_info(result)
        else:
            # 显示原始帧
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image(rgb_image)

    # ========== 视频文件操作 ==========

    def load_video(self):
        """加载视频文件"""
        self.stop_camera()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*)"
        )

        if file_path:
            self.current_video_path = file_path
            self.status_bar.showMessage(f"正在加载视频: {os.path.basename(file_path)}")

            # 验证视频是否能正常打开
            try:
                cap = cv2.VideoCapture(file_path)

                if not cap.isOpened():
                    QMessageBox.warning(self, "错误",
                        f"无法打开视频文件\n\n请检查：\n"
                        f"1. 文件路径是否正确\n"
                        f"2. 视频文件是否损坏\n"
                        f"3. 是否安装了相应的解码器\n"
                        f"4. 视频编码格式是否被支持")
                    cap.release()
                    self.current_video_path = None
                    return

                # 获取视频信息
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # 尝试读取第一帧作为预览
                ret, frame = cap.read()

                if not ret or frame is None:
                    QMessageBox.warning(self, "错误",
                        "无法读取视频帧\n\n视频文件可能损坏或编码格式不支持")
                    cap.release()
                    self.current_video_path = None
                    return

                # 显示第一帧作为预览
                self.current_image = frame
                self.display_image(frame)

                # 更新状态和检测按钮
                self.start_detect_button.setEnabled(True)

                # 显示视频信息
                video_info = f"视频: {os.path.basename(file_path)} | "
                video_info += f"分辨率: {width}x{height} | "
                video_info += f"帧数: {frame_count} | "
                video_info += f"帧率: {fps:.2f}fps"
                self.status_bar.showMessage(video_info)

                self.detection_info_label.setText(
                    f"视频已加载 | 分辨率: {width}x{height} | 总帧数: {frame_count}"
                )

                cap.release()

            except Exception as e:
                QMessageBox.critical(self, "错误",
                    f"加载视频时发生错误:\n\n{str(e)}\n\n{type(e).__name__}")
                self.current_video_path = None
                print(f"加载视频错误: {e}")
                import traceback
                traceback.print_exc()

    def process_video_file(self):
        """处理视频文件"""
        if not hasattr(self, 'current_video_path') or not self.current_video_path:
            QMessageBox.warning(self, "警告", "请先加载视频文件")
            return

        # 输出路径
        output_name = f"output_{os.path.basename(self.current_video_path)}"
        output_path = Config.get_output_path(output_name)

        # 确认对话框
        reply = QMessageBox.question(
            self, "确认处理",
            f"即将处理视频文件:\n{os.path.basename(self.current_video_path)}\n\n"
            f"输出文件: {output_name}\n\n"
            f"处理过程可能需要较长时间，是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        # 进度回调
        last_update_time = 0

        def progress_callback(current, total, frame):
            nonlocal last_update_time

            # 限制UI更新频率，避免卡顿（每10帧更新一次）
            current_time = current
            if current - last_update_time >= 10 or current == total:
                self.progress_bar.setValue(int(current / total * 100))
                self.status_bar.showMessage(f"处理进度: {current}/{total} ({current/total*100:.1f}%)")
                if frame is not None:
                    self.display_image(frame)
                last_update_time = current

        # 执行检测
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_detect_button.setEnabled(False)
        self.stop_detect_button.setEnabled(True)
        self.status_bar.showMessage("正在处理视频...")

        try:
            result = self.detector.detect_video_file(
                self.current_video_path,
                Config.CONF_THRESHOLD,
                Config.IOU_THRESHOLD,
                output_path,
                progress_callback
            )

            # 处理结果
            self.progress_bar.setVisible(False)
            self.start_detect_button.setEnabled(True)
            self.stop_detect_button.setEnabled(False)

            if result:
                self.save_detection_record('video', self.current_video_path, result)
                self.update_statistics()

                # 显示详细结果
                result_text = (
                    f"视频处理完成！\n\n"
                    f"总帧数: {result['total_frames']}\n"
                    f"已处理: {result['processed_frames']}\n"
                    f"检测到异物: {result['total_objects']} 个\n"
                    f"平均置信度: {result['avg_confidence']:.2%}\n"
                    f"最高置信度: {result['max_confidence']:.2%}\n\n"
                    f"输出文件: {output_path}"
                )
                QMessageBox.information(self, "处理完成", result_text)
            else:
                QMessageBox.warning(self, "错误",
                    "视频处理失败\n\n可能原因:\n"
                    "1. 视频文件损坏\n"
                    "2. 编码格式不支持\n"
                    "3. 内存不足")
                self.status_bar.showMessage("视频处理失败")

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.start_detect_button.setEnabled(True)
            self.stop_detect_button.setEnabled(False)

            error_msg = f"视频处理失败:\n\n{str(e)}\n\n{type(e).__name__}"
            QMessageBox.warning(self, "错误", error_msg)
            print(f"视频检测错误: {e}")
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage("处理出错")

    # ========== 检测控制 ==========

    def start_detection(self):
        """开始检测"""
        self.is_detecting = True
        self.start_detect_button.setEnabled(False)
        self.stop_detect_button.setEnabled(True)

        if self.current_image is not None:
            # 检测当前图片
            result = self.detector.detect_image(self.current_image, Config.CONF_THRESHOLD, Config.IOU_THRESHOLD)
            if result:
                self.display_image(result['image'])
                self.update_detection_info(result)
                self.save_detection_record('image', self.current_image_path or 'unknown', result)
                self.update_statistics()

        elif hasattr(self, 'current_video_path') and self.current_video_path:
            # 处理视频文件
            self.process_video_file()

        self.status_bar.showMessage("检测中...")

    def stop_detection(self):
        """停止检测"""
        self.is_detecting = False
        self.start_detect_button.setEnabled(True)
        self.stop_detect_button.setEnabled(False)
        self.status_bar.showMessage("检测已停止")

    def update_detection_info(self, result):
        """更新检测信息显示"""
        info_text = f"检测到: {result['count']} 个异物 | 最高置信度: {result['max_conf']:.2f} | 平均置信度: {result['avg_conf']:.2f}"
        self.detection_info_label.setText(info_text)

    # ========== 数据库操作 ==========

    def save_detection_record(self, detection_type, source, result):
        """保存检测记录到数据库"""
        try:
            self.database.add_record(
                user_id=self.user_info['id'],
                detection_type=detection_type,
                source=source,
                object_count=result['count'],
                max_conf=result['max_conf'],
                avg_conf=result['avg_conf']
            )
        except Exception as e:
            print(f"保存记录失败: {e}")

    def update_statistics(self):
        """更新统计信息"""
        try:
            # 总体统计
            stats = self.statistics.get_overall_statistics()
            overall_text = (
                f"检测次数: {stats['total_detections']}\n"
                f"异物总数: {stats['total_objects']}\n"
                f"平均置信度: {stats['avg_confidence']:.2%}\n"
                f"最高置信度: {stats['max_confidence']:.2%}\n"
                f"使用人数: {stats['user_count']}"
            )
            self.overall_stats_label.setText(overall_text)

            # 实时统计
            self.realtime_stats_label.setText(
                f"检测次数: {stats['total_detections']}\n"
                f"异物总数: {stats['total_objects']}\n"
                f"平均置信度: {stats['avg_confidence']:.2%}"
            )

            # 历史记录
            self.update_records_table()

        except Exception as e:
            print(f"更新统计失败: {e}")

    def update_records_table(self):
        """更新历史记录表格"""
        try:
            records = self.statistics.get_recent_records(Config.RECENT_RECORDS_LIMIT)

            self.records_table.setRowCount(len(records))

            for row, record in enumerate(records):
                # 时间
                time_item = QTableWidgetItem(str(record[7]))
                self.records_table.setItem(row, 0, time_item)

                # 类型
                type_item = QTableWidgetItem(record[2])
                self.records_table.setItem(row, 1, type_item)

                # 来源
                source_item = QTableWidgetItem(str(record[3][:30] + '...' if len(str(record[3])) > 30 else record[3]))
                self.records_table.setItem(row, 2, source_item)

                # 异物数
                count_item = QTableWidgetItem(str(record[4]))
                self.records_table.setItem(row, 3, count_item)

                # 置信度
                conf_item = QTableWidgetItem(f"{record[6]:.2%}")
                self.records_table.setItem(row, 4, conf_item)

        except Exception as e:
            print(f"更新记录表失败: {e}")

    # ========== 其他 ==========

    def logout(self):
        """退出登录"""
        self.stop_camera()
        self.close()
        # 重新显示登录窗口
        if hasattr(self, 'parent') and hasattr(self.parent(), 'login_window'):
            self.parent().login_window.show()
        else:
            # 如果没有父窗口，重新创建登录窗口
            import main
            login_window = main.LoginWindow()
            login_window.show()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_camera()
        try:
            self.database.close()
        except:
            pass
        event.accept()
