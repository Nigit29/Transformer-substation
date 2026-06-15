"""
配置文件 - 存储应用程序配置
"""
import os


class Config:
    # ========== 路径配置 ==========
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, 'substation_detection.db')
    MODEL_PATH = os.path.join(BASE_DIR, 'biandianzhan/weights/best.pt')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

    # ========== 检测参数 ==========
    CONF_THRESHOLD = 0.60      # 置信度阈值（提高以减少误检）
    IOU_THRESHOLD = 0.45       # IOU阈值
    IMAGE_SIZE = 640           # 图片尺寸

    # ========== 摄像头配置 ==========
    CAMERA_INDEX = 0           # 摄像头索引 (如果0不工作，尝试1、2等)
    CAMERA_FPS = 30            # 摄像头帧率
    CAMERA_WIDTH = 640         # 摄像头宽度
    CAMERA_HEIGHT = 480        # 摄像头高度
    AUTO_DETECT_CAMERA = True  # 自动检测可用摄像头

    # ========== 界面配置 ==========
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    DISPLAY_WIDTH = 800        # 显示区域宽度
    DISPLAY_HEIGHT = 600       # 显示区域高度

    # ========== 统计配置 ==========
    RECENT_RECORDS_LIMIT = 10  # 显示最近的记录数量
    STATISTICS_DAYS = 7        # 统计最近多少天

    # ========== 用户配置 ==========
    DEFAULT_ADMIN_USERNAME = 'admin'
    DEFAULT_ADMIN_PASSWORD = 'admin123'

    # ========== 支持的图片格式 ==========
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # ========== 支持的视频格式 ==========
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    @classmethod
    def ensure_output_dir(cls):
        """确保输出目录存在"""
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)
        return cls.OUTPUT_DIR

    @classmethod
    def get_output_path(cls, filename):
        """获取输出文件路径"""
        cls.ensure_output_dir()
        return os.path.join(cls.OUTPUT_DIR, filename)
