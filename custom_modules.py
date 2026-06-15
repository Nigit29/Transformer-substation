"""
自定义YOLOv8模块：集成SE注意力机制
用于ultralytics YOLOv8模型注册
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, Conv


class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力模块"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局池化压缩空间维度
        y = self.avg_pool(x).view(b, c)
        # Excitation: 学习通道重要性权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 对原始特征进行通道重标定
        return x * y.expand_as(x)


class C2f_SE(nn.Module):
    """集成SE注意力的C2f模块

    在原始C2f模块基础上，对每个分支输出应用SE注意力，
    增强对小目标相关特征通道的关注，抑制背景噪声。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = nn.Conv2d(c1, 2 * c2, 1, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * c2, c2, 1, 1, bias=False)

        # SE注意力模块
        self.se = SELayer(c2, reduction=16)

        # 多分支结构
        self.m = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c2, self.c, 1, bias=False),
                nn.Conv2d(self.c, c2, 1, bias=False)
            ) for _ in range(n)
        )

        # 是否使用shortcut连接
        self.shortcut = shortcut

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            输出特征图 [B, C2, H, W]
        """
        x1 = self.cv1(x)
        # 分割为两个部分
        x2, x3 = x1.chunk(2, dim=1)

        # 收集所有分支输出
        y = [x2, x3]

        # 多分支处理，每个分支应用SE注意力
        for m in self.m:
            branch_out = m(x2)
            # 应用SE注意力增强小目标特征
            branch_out = self.se(branch_out)
            y.append(branch_out)

        # Concatenate所有分支
        y = torch.cat(y, dim=1)

        # 最终输出
        out = self.cv2(y)

        # 如果使用shortcut连接且维度匹配
        if self.shortcut and out.shape == x.shape:
            out = out + x

        return out


class SE_Bottleneck(nn.Module):
    """带SE注意力的Bottleneck模块"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False, groups=g)
        self.cv3 = nn.Conv2d(c2, c2, 1, 1, bias=False)

        # SE注意力模块
        self.se = SELayer(c2, reduction=16)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)

        # 应用SE注意力
        x2 = self.se(x2)

        x3 = self.cv3(x2)

        if self.add:
            x3 = x3 + x

        return x3


def register_custom_modules():
    """注册自定义模块到ultralytics"""
    try:
        from ultralytics.nn.modules import __all__

        # 添加C2f_SE到模块列表
        if 'C2f_SE' not in __all__:
            __all__.append('C2f_SE')

        # 添加模块到ultralytics.nn.modules命名空间
        import ultralytics.nn.modules as um
        um.C2f_SE = C2f_SE
        um.SELayer = SELayer

        print("自定义模块C2f_SE已成功注册")
        return True
    except Exception as e:
        print(f"模块注册失败: {e}")
        return False


# 自动注册模块
if __name__ != '__main__':
    register_custom_modules()