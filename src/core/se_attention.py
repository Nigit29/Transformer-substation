"""
SE (Squeeze-and-Excitation)注意力模块实现
用于YOLOv8骨干网络中强化小目标特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Squeeze: 全局池化
        y = self.avg_pool(x).view(b, c)
        # Excitation: 学习通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 重新加权
        return x * y.expand_as(x)


class SE_C2f(nn.Module):
    """集成SE注意力的C2f模块"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
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
        x1 = self.cv1(x)

        # 主分支
        x2 = x1[0]
        x3 = x1[1]

        # 多分支处理
        y = []
        y.append(x2)
        y.append(x3)

        for m in self.m:
            # 每个分支应用SE注意力
            branch_out = m(x2)
            branch_out = self.se(branch_out)  # 应用SE注意力
            y.append(branch_out)

        # Concatenate所有分支
        y = torch.cat(y, 1)

        # 最终输出
        out = self.cv2(y)

        # 如果使用shortcut连接
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


class SE_CSPDarknet(nn.Module):
    """带SE注意力的CSPDarknet骨干网络"""

    def __init__(self, depths=[3, 3, 6, 3], widths=[64, 128, 256, 512]):
        super().__init__()

        # 第一层
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(),
            nn.Conv2d(widths[0], widths[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU()
        )

        # 中间层
        self.stages = nn.ModuleList()
        for i in range(4):
            layer = nn.ModuleList()

            # Bottleneck模块
            for j in range(depths[i]):
                layer.append(SE_Bottleneck(
                    widths[i] if j == 0 else widths[i] // 2,
                    widths[i] // 2
                ))

            self.stages.append(layer)

        # 输出卷积
        self.out_convs = nn.ModuleList([
            nn.Conv2d(widths[i], widths[i], 1, 1, bias=False)
            for i in range(4)
        ])

    def forward(self, x):
        x = self.stem(x)

        features = []
        x_prev = x

        for i, stage in enumerate(self.stages):
            x = x_prev
            for module in stage:
                x = module(x)
            x_prev = x
            features.append(self.out_convs[i](x))

        return features[::-1]  # 返回逆序特征图（大->小）