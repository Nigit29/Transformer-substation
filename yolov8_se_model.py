"""
YOLOv8-SE 模型完整实现
基于PyTorch实现带SE注意力的YOLOv8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== SE注意力模块 ====================
class SELayer(nn.Module):
    """
    Squeeze-and-Excitation注意力模块

    作用：
    1. 通过全局池化压缩空间维度
    2. 学习通道间的重要性权重
    3. 重新加权特征，增强小目标相关通道
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 通道注意力网络：降维 -> 激活 -> 升维
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            重新加权后的特征图 [B, C, H, W]
        """
        b, c, _, _ = x.size()

        # Squeeze: 全局池化压缩空间维度
        y = self.avg_pool(x).view(b, c)

        # Excitation: 学习通道重要性权重
        y = self.fc(y).view(b, c, 1, 1)

        # Scale: 对原始特征进行通道重标定
        return x * y.expand_as(x)


# ==================== 卷积模块 ====================
class Conv(nn.Module):
    """标准卷积 + BN + SiLU激活"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None):
    """自动计算padding以保持输入输出尺寸一致"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ==================== C2f-SE模块 ====================
class C2f_SE(nn.Module):
    """
    带SE注意力的C2f模块（核心改进）

    在原始C2f基础上，对每个分支输出应用SE注意力，
    专门针对小目标特征增强。

    结构：
    - 输入特征
    - 分割为两个分支
    - 多个Bottleneck分支处理
    - 每个分支应用SE注意力
    - Concatenate + 卷积输出
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数

        # 1x1卷积，将输入扩展为2倍通道
        self.cv1 = Conv(c1, 2 * c2, 1, 1)

        # 输出卷积
        self.cv2 = Conv((2 + n) * c2, c2, 1)

        # 多个Bottleneck分支
        self.m = nn.ModuleList(
            Bottleneck(c2, c2, shortcut, g, e=1.0) for _ in range(n)
        )

        # SE注意力模块（在每个分支后应用）
        self.se = SELayer(c2, reduction=16)

    def forward(self, x):
        """前向传播"""
        x1 = self.cv1(x)
        x1 = x1.chunk(2, 1)  # 分割为两个部分

        y = list(x1)  # 保存两个分支

        # 处理每个Bottleneck分支并应用SE注意力
        for i, module in enumerate(self.m):
            # Bottleneck处理
            y_i = module(y[-2])  # 使用第二个分支作为输入

            # 应用SE注意力增强小目标特征
            y_i = self.se(y_i)

            y.append(y_i)

        # 拼接所有分支
        y = torch.cat(y, 1)

        # 输出卷积
        return self.cv2(y)


# ==================== Bottleneck模块 ====================
class Bottleneck(nn.Module):
    """标准Bottleneck模块"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ==================== SPPF模块 ====================
class SPPF(nn.Module):
    """空间金字塔池化模块"""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


# ==================== 检测头模块 ====================
class Detect(nn.Module):
    """YOLOv8检测头"""

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # 输出通道数

        self.stride = torch.zeros(self.nl)  # 步幅
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)

        # 分支输出
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

        for i in range(self.nl):
            self.cv2.append(nn.Conv2d(ch[i], c2, 3, 1, 1))
            self.cv3.append(nn.Conv2d(c2, c3, 3, 1, 1))

        # DFL（Distribution Focal Loss）
        self.dfl = nn.Conv2d(c3, 4 * self.reg_max, 1, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化检测头权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """前向传播"""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = self.cv3[i](self.cv2[i](x[i]))

        # 整合输出
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        return x_cat


# ==================== 完整YOLOv8-SE模型 ====================
class YOLOv8_SE(nn.Module):
    """
    YOLOv8-SE 完整模型

    改进点：
    1. 骨干网络中使用C2f_SE模块
    2. 检测头中也使用C2f_SE模块
    3. SE注意力增强小目标特征提取
    """

    def __init__(self, nc=1):
        super(YOLOv8_SE, self).__init__()
        self.nc = nc

        # ============ 骨干网络 ============
        # P1
        self.stem = nn.Sequential(
            Conv(3, 64, 3, 2),
            Conv(64, 128, 3, 2),
        )

        # P2
        self.dark2 = nn.Sequential(
            C2f_SE(128, 128, n=3, shortcut=True),
            Conv(128, 256, 3, 2),
        )

        # P3
        self.dark3 = nn.Sequential(
            C2f_SE(256, 256, n=6, shortcut=True),
            Conv(256, 512, 3, 2),
        )

        # P4
        self.dark4 = nn.Sequential(
            C2f_SE(512, 512, n=6, shortcut=True),
            Conv(512, 1024, 3, 2),
        )

        # P5
        self.dark5 = nn.Sequential(
            C2f_SE(1024, 1024, n=3, shortcut=True),
            SPPF(1024, 1024, 5),
        )

        # ============ 检测头 ============
        # 逆序特征融合
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.detect_reduce0 = Conv(1024, 512, 1, 1)
        self.detect_c2f0 = C2f_SE(1024, 512, n=3, shortcut=False)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.detect_reduce1 = Conv(512, 256, 1, 1)
        self.detect_c2f1 = C2f_SE(512, 256, n=3, shortcut=False)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.detect_reduce2 = Conv(256, 128, 1, 1)
        self.detect_c2f2 = C2f_SE(256, 128, n=3, shortcut=False)

        # 下采样路径
        self.downsample0 = Conv(128, 128, 3, 2)
        self.detect_c2f3 = C2f_SE(256, 256, n=3, shortcut=False)

        self.downsample1 = Conv(256, 256, 3, 2)
        self.detect_c2f4 = C2f_SE(512, 512, n=3, shortcut=False)

        # 检测头
        self.detect = Detect(nc=nc, ch=(128, 256, 512))

    def forward(self, x):
        """前向传播"""

        # 骨干网络
        x = self.stem(x)  # [B, 128, H/4, W/4]
        x = self.dark2(x)  # [B, 256, H/8, W/8]
        p3 = self.dark3(x)  # [B, 512, H/16, W/16]
        p4 = self.dark4(p3)  # [B, 1024, H/32, W/32]
        p5 = self.dark5(p4)  # [B, 1024, H/32, W/32]

        # 检测头 - 上采样路径
        # P5 -> P4
        x = self.detect_reduce0(p5)
        x = self.upsample0(x)
        x = torch.cat([x, p4], dim=1)
        x = self.detect_c2f0(x)  # [B, 512, H/16, W/16]

        # P4 -> P3
        x = self.detect_reduce1(x)
        x = self.upsample1(x)
        x = torch.cat([x, p3], dim=1)
        x = self.detect_c2f1(x)  # [B, 256, H/8, W/8]

        # P3 -> P2
        x = self.detect_reduce2(x)
        x = self.upsample2(x)
        x = torch.cat([x, self.dark2[:1](x)], dim=1)
        x = self.detect_c2f2(x)  # [B, 128, H/4, W/4]

        # 下采样路径
        # P2 -> P3
        p2_out = x
        x = self.downsample0(x)
        x = torch.cat([x, self.detect_c2f1[:2](self.detect_c2f1[-1].cv1(self.detect_c2f1[-1].cv2(torch.cat([self.detect_reduce1(self.detect_c2f0[-1](self.detect_c2f0[-1](self.detect_reduce0(p5))), self.detect_reduce0(p5)], dim=1))))), dim=1)
        x = self.detect_c2f3(x)  # [B, 256, H/8, W/8]

        # P3 -> P4
        x = self.downsample1(x)
        x = torch.cat([x, self.detect_c2f0[:2](self.detect_c2f0[-1](self.detect_c2f0[-1](self.detect_reduce0(p5)))), dim=1)
        x = self.detect_c2f4(x)  # [B, 512, H/16, W/16]

        # 检测输出
        return self.detect([p2_out, x, self.detect_c2f4[:2](self.detect_c2f4[-1](self.detect_c2f4[-1](torch.cat([self.downsample1(self.detect_c2f3[-1](self.detect_c2f3[-1](torch.cat([self.downsample0(p2_out), self.detect_c2f1[:2](self.detect_c2f1[-1](self.detect_c2f1[-1](torch.cat([self.detect_reduce1(self.detect_c2f0[-1](self.detect_c2f0[-1](self.detect_reduce0(p5)))), self.detect_reduce0(p5)], dim=1))))), dim=1))), dim=1])))

    def load_state_dict_ultralytics(self, ultralytics_model):
        """从ultralytics模型加载权重（兼容）"""
        # 这里需要实现权重映射
        pass


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("测试YOLOv8-SE模型...")

    # 创建模型
    model = YOLOv8_SE(nc=1)

    # 测试前向传播
    test_input = torch.randn(1, 3, 640, 640)
    output = model(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 统计SE模块数量
    se_count = sum(1 for m in model.modules() if isinstance(m, SELayer))
    print(f"SE模块数量: {se_count}")

    print("\n模型结构:")
    print(model)

    print("\n✓ 测试通过")
