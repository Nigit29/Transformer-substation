# 变电站检测项目 (Substation Detection)

基于 YOLOv8 和 SE 注意力机制的变电站设备目标检测系统。

## 项目结构

```
biandianzhan/
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── main.py            # 主程序入口
│   ├── train.py           # 训练脚本
│   ├── train_se_final.py  # SE 注意力机制训练脚本
│   ├── add_negative_samples.py  # 负样本添加脚本
│   ├── batch_detect.py    # 批量检测脚本
│   ├── test_video.py      # 视频测试脚本
│   ├── statistics.py      # 统计分析脚本
│   ├── core/              # 核心模块
│   │   ├── __init__.py
│   │   ├── detector.py    # 基础检测器
│   │   ├── detector_se.py # SE 注意力检测器
│   │   ├── se_attention.py # SE 注意力机制
│   │   ├── yolov8_se_model.py # YOLOv8-SE 模型
│   │   └── custom_modules.py  # 自定义模块
│   ├── ui/                # UI 界面模块
│   │   ├── __init__.py
│   │   └── main_window.py # 主窗口界面
│   └── utils/             # 工具模块
│       ├── __init__.py
│       ├── config.py      # 配置管理
│       ├── database.py    # 数据库操作
│       └── user_manager.py # 用户管理
├── models/                # 模型文件目录
│   ├── yolov8n.pt         # YOLOv8n 预训练权重
│   └── yolov8n_se.yaml    # YOLOv8-SE 模型配置
├── data/                  # 数据目录
│   ├── args.yaml          # 训练参数配置
│   └── results/           # 训练结果和图表
├── docs/                  # 文档目录
├── tests/                 # 测试目录
├── .gitignore
└── README.md
```

## 功能特性

- ✅ YOLOv8 目标检测
- ✅ SE (Squeeze-and-Excitation) 注意力机制改进
- ✅ 图形化用户界面
- ✅ 批量图片/视频检测
- ✅ 训练结果可视化
- ✅ 数据库存储检测结果
- ✅ 用户权限管理

## 安装依赖

```bash
pip install ultralytics PyQt5 opencv-python numpy pandas torch torchvision
```

## 快速开始

### 运行主程序

```bash
python src/main.py
```

### 训练模型

```bash
# 基础训练
python src/train.py

# SE 注意力机制训练
python src/train_se_final.py
```

### 批量检测

```bash
python src/batch_detect.py
```

## 配置说明

编辑 `src/utils/config.py` 修改以下配置：

- 模型路径
- 数据集路径
- 训练参数
- 数据库连接

## 模型性能

| 模型 | mAP | 参数量 | 推理速度 |
|------|-----|--------|----------|
| YOLOv8n | - | 3.2M | - |
| YOLOv8n-SE | - | - | - |

## 许可证

MIT License

## 作者

Ni (2598842676@qq.com)
