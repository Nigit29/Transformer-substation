"""
视频加载测试脚本
帮助诊断视频加载问题
"""
import cv2
import sys
import os


def test_video_file(video_path):
    """测试视频文件能否正常加载"""

    print("="*60)
    print("视频文件测试")
    print("="*60)

    # 1. 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"✗ 文件不存在: {video_path}")
        return False
    print(f"✓ 文件存在: {video_path}")

    # 2. 检查文件大小
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"✓ 文件大小: {file_size:.2f} MB")

    # 3. 检查文件扩展名
    ext = os.path.splitext(video_path)[1].lower()
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    if ext in supported_formats:
        print(f"✓ 文件格式: {ext} (支持)")
    else:
        print(f"⚠ 文件格式: {ext} (可能不支持)")
    print()

    # 4. 尝试打开视频（使用不同后端）
    print("尝试打开视频...")

    # Windows优先使用DirectShow
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == 'win32' else [cv2.CAP_ANY]

    cap = None
    used_backend = None

    for backend in backends:
        try:
            backend_name = {
                cv2.CAP_DSHOW: "DirectShow",
                cv2.CAP_MSMF: "Media Foundation",
                cv2.CAP_ANY: "Default"
            }.get(backend, f"Unknown({backend})")

            cap = cv2.VideoCapture(video_path, backend)
            if cap.isOpened():
                used_backend = backend_name
                print(f"✓ 成功打开视频 (后端: {backend_name})")
                break
            else:
                cap.release()
        except Exception as e:
            print(f"✗ 后端 {backend_name} 失败: {e}")
            if cap:
                cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        print("\n✗ 无法打开视频文件")
        print("\n可能的解决方案:")
        print("1. 安装K-Lite Codec Pack或类似的解码器包")
        print("2. 使用FFmpeg或其他工具转换视频格式")
        print("3. 尝试使用不同格式的视频文件（推荐MP4/H.264）")
        return False

    # 5. 获取视频信息
    print("\n视频信息:")
    print(f"  宽度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"  高度: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  帧率: {cap.get(cv2.CAP_PROP_FPS):.2f} fps")
    print(f"  总帧数: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"  时长: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS):.2f} 秒")

    # 6. 测试读取帧
    print("\n测试读取帧...")

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("✗ 无法读取第一帧")
        cap.release()
        return False
    print(f"✓ 成功读取第一帧 (尺寸: {frame.shape})")

    # 读取第10帧（测试seek）
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    if ret:
        print(f"✓ 成功读取第10帧")
    else:
        print("⚠ 无法跳转到第10帧（可能是编码问题）")

    # 尝试读取多帧（测试稳定性）
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    success_count = 0
    for i in range(min(100, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        frame_count += 1
        if ret:
            success_count += 1

    print(f"✓ 读取 {frame_count} 帧，成功 {success_count} 帧 (成功率: {success_count/frame_count*100:.1f}%)")

    # 7. 测试视频编码器
    print("\n测试视频编码器...")

    fourcc_list = [
        ('mp4v', 'MP4V'),
        ('XVID', 'XVID'),
        ('MJPG', 'MJPEG'),
        ('avc1', 'H.264'),
    ]

    for name, desc in fourcc_list:
        fourcc = cv2.VideoWriter_fourcc(*name)
        test_path = "test_output.mp4"
        try:
            writer = cv2.VideoWriter(test_path, fourcc, 30.0, (640, 480))
            if writer.isOpened():
                print(f"✓ 编码器 {name} ({desc}) 可用")
                writer.release()
                if os.path.exists(test_path):
                    os.remove(test_path)
            else:
                print(f"✗ 编码器 {name} ({desc}) 不可用")
        except Exception as e:
            print(f"✗ 编码器 {name} ({desc}) 错误: {e}")

    # 8. 结论
    print("\n" + "="*60)
    print("测试结论:")
    print("="*60)

    if success_count == frame_count and frame_count > 0:
        print("✓ 视频文件可以正常读取")
        print("✓ 应该可以在程序中正常使用")
        return True
    else:
        print("✗ 视频文件存在问题")
        print("\n建议:")
        print("1. 视频可能使用了OpenCV不支持的编码")
        print("2. 尝试使用FFmpeg转换视频:")
        print(f"   ffmpeg -i \"{video_path}\" -c:v libx264 -c:a aac output.mp4")
        print("3. 或者使用在线转换工具转换为MP4格式")
        return False

    cap.release()


def main():
    """主函数"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    # 选择视频文件
    video_path = filedialog.askopenfilename(
        title="选择要测试的视频文件",
        filetypes=[
            ("视频文件", "*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv"),
            ("所有文件", "*.*")
        ]
    )

    if video_path:
        test_video_file(video_path)
    else:
        print("未选择文件")

    input("\n按回车键退出...")


if __name__ == '__main__':
    main()
