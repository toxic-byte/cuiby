#!/usr/bin/env python3
"""
生成第四章适配器结构图：
1. 单个适配器模块（ResMLP）的内部结构
2. 适配器在Transformer层中的注入位置
3. 双适配器并行运行机制
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'STHeiti', 'PingFang SC', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2, 1.5]})

# ===== 颜色定义 =====
color_input = '#E3F2FD'       # 浅蓝 - 输入
color_linear = '#BBDEFB'      # 蓝色 - 线性层
color_act = '#C8E6C9'         # 绿色 - 激活函数
color_norm = '#FFF9C4'        # 黄色 - 归一化
color_adapter = '#E1BEE7'     # 紫色 - 适配器模块
color_attn = '#FFCCBC'        # 橙色 - 注意力
color_ffn = '#D1C4E9'         # 淡紫 - FFN
color_add = '#F0F4C3'         # 淡黄绿 - 加法
color_frozen = '#CFD8DC'      # 灰色 - 冻结
color_trainable = '#A5D6A7'   # 绿色 - 可训练
color_border = '#37474F'

def draw_box(ax, x, y, w, h, text, facecolor, fontsize=10, edgecolor=color_border, linewidth=1.5, alpha=0.9, textcolor='black', fontstyle='normal', fontweight='normal'):
    """绘制带文字的圆角矩形"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontstyle=fontstyle, fontweight=fontweight)

def draw_arrow(ax, x1, y1, x2, y2, color='#455A64', style='->', lw=1.5):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ========================================
# (a) 适配器模块内部结构
# ========================================
ax1 = axes[0]
ax1.set_xlim(-2, 2)
ax1.set_ylim(-1, 10)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('(a) 适配器模块 (ResMLP)', fontsize=13, fontweight='bold', pad=15)

# 从下到上绘制
bw, bh = 2.2, 0.7
cx = 0

# 输入 x
draw_box(ax1, cx, 0, bw, bh, r'输入 $\mathbf{x}$', color_input, fontsize=11)

# W_down 降维
draw_arrow(ax1, cx, 0.35, cx, 1.15)
draw_box(ax1, cx, 1.5, bw, bh, r'$\mathbf{W}_{down}$ (降维)', color_linear, fontsize=10)

# ReLU
draw_arrow(ax1, cx, 1.85, cx, 2.65)
draw_box(ax1, cx, 3.0, bw, bh, 'ReLU', color_act, fontsize=11)

# W_up 升维
draw_arrow(ax1, cx, 3.35, cx, 4.15)
draw_box(ax1, cx, 4.5, bw, bh, r'$\mathbf{W}_{up}$ (升维)', color_linear, fontsize=10)

# LN
draw_arrow(ax1, cx, 4.85, cx, 5.65)
draw_box(ax1, cx, 6.0, bw, bh, 'LayerNorm', color_norm, fontsize=11)

# 残差连接 (+)
draw_arrow(ax1, cx, 6.35, cx, 7.15)
draw_box(ax1, cx, 7.5, 1.0, 0.6, '+', '#E8F5E9', fontsize=14, fontweight='bold')

# 残差连接线（从输入绕到加号）
ax1.annotate('', xy=(-0.5, 7.2), xytext=(-0.5, 0),
             arrowprops=dict(arrowstyle='-', color='#F44336', lw=2, linestyle='--'))
ax1.annotate('', xy=(0, 7.2), xytext=(-0.5, 7.2),
             arrowprops=dict(arrowstyle='->', color='#F44336', lw=2, linestyle='--'))
ax1.text(-1.3, 3.5, '残差\n连接', fontsize=9, color='#F44336', ha='center', va='center', rotation=90)

# 输出
draw_arrow(ax1, cx, 7.8, cx, 8.65)
draw_box(ax1, cx, 9.0, bw, bh, r'输出 $\text{Adapter}(\mathbf{x})$', color_input, fontsize=10)

# 维度标注
ax1.text(1.5, 1.5, r'$d \to d_b$', fontsize=9, color='#666666', ha='left', va='center')
ax1.text(1.5, 4.5, r'$d_b \to d$', fontsize=9, color='#666666', ha='left', va='center')
ax1.text(1.5, 0, r'$d=1280$', fontsize=8, color='#999999', ha='left', va='center')
ax1.text(1.5, 3.0, r'$d_b=640$', fontsize=8, color='#999999', ha='left', va='center')


# ========================================
# (b) 适配器在Transformer层中的注入
# ========================================
ax2 = axes[1]
ax2.set_xlim(-3, 3)
ax2.set_ylim(-1, 10)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('(b) Transformer层中的注入位置', fontsize=13, fontweight='bold', pad=15)

bw2 = 2.8
cx2 = 0

# 输入 X
draw_box(ax2, cx2, 0, bw2, 0.6, r'输入 $\mathbf{X}$', color_input, fontsize=10)

# LN
draw_arrow(ax2, cx2, 0.3, cx2, 0.9)
draw_box(ax2, cx2, 1.2, bw2, 0.5, 'LayerNorm', color_norm, fontsize=9)

# MHA
draw_arrow(ax2, cx2, 1.45, cx2, 1.95)
draw_box(ax2, cx2, 2.3, bw2, 0.6, '多头自注意力 (MHA)', color_attn, fontsize=9)

# Adapter_attn
draw_arrow(ax2, cx2, 2.6, cx2, 3.1)
draw_box(ax2, cx2, 3.5, bw2, 0.6, r'$\text{Adapter}_{attn}$', color_adapter, fontsize=10, fontweight='bold')

# 残差 +
draw_arrow(ax2, cx2, 3.8, cx2, 4.3)
draw_box(ax2, cx2, 4.6, 0.6, 0.4, '+', '#E8F5E9', fontsize=12, fontweight='bold')

# 残差连接线
ax2.annotate('', xy=(-0.3, 4.4), xytext=(-0.3, 0),
             arrowprops=dict(arrowstyle='-', color='#F44336', lw=1.5, linestyle='--'))
ax2.annotate('', xy=(0, 4.4), xytext=(-0.3, 4.4),
             arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5, linestyle='--'))

# X' 输出
draw_arrow(ax2, cx2, 4.8, cx2, 5.2)
draw_box(ax2, cx2, 5.5, bw2*0.6, 0.4, r"$\mathbf{X}'$", '#E8EAF6', fontsize=10)

# LN
draw_arrow(ax2, cx2, 5.7, cx2, 6.1)
draw_box(ax2, cx2, 6.4, bw2, 0.5, 'LayerNorm', color_norm, fontsize=9)

# FFN
draw_arrow(ax2, cx2, 6.65, cx2, 7.05)
draw_box(ax2, cx2, 7.35, bw2, 0.5, '前馈网络 (FFN)', color_ffn, fontsize=9)

# Adapter_ffn
draw_arrow(ax2, cx2, 7.6, cx2, 7.95)
draw_box(ax2, cx2, 8.3, bw2, 0.6, r'$\text{Adapter}_{ffn}$', color_adapter, fontsize=10, fontweight='bold')

# 残差 +
draw_arrow(ax2, cx2, 8.6, cx2, 8.9)
draw_box(ax2, cx2, 9.15, 0.6, 0.4, '+', '#E8F5E9', fontsize=12, fontweight='bold')

# 残差连接线
ax2.annotate('', xy=(-0.3, 8.95), xytext=(-0.3, 5.5),
             arrowprops=dict(arrowstyle='-', color='#F44336', lw=1.5, linestyle='--'))
ax2.annotate('', xy=(0, 8.95), xytext=(-0.3, 8.95),
             arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5, linestyle='--'))

# 标注注入位置
ax2.annotate('注入位置①', xy=(1.5, 3.5), fontsize=8, color='#7B1FA2',
             ha='left', fontweight='bold')
ax2.annotate('注入位置②', xy=(1.5, 8.3), fontsize=8, color='#7B1FA2',
             ha='left', fontweight='bold')


# ========================================
# (c) 双适配器并行机制
# ========================================
ax3 = axes[2]
ax3.set_xlim(-4.5, 4.5)
ax3.set_ylim(-1, 10)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('(c) 双适配器并行融合', fontsize=13, fontweight='bold', pad=15)

# 输入
draw_box(ax3, 0, 0, 2.5, 0.6, r'输入 $\mathbf{X}$', color_input, fontsize=10)

# 分叉
draw_arrow(ax3, -0.5, 0.3, -2, 1.2)
draw_arrow(ax3, 0.5, 0.3, 2, 1.2)

# Adapter_0 (冻结)
draw_box(ax3, -2, 2.2, 3.0, 1.8, '', color_frozen, fontsize=10, linewidth=2)
draw_box(ax3, -2, 2.8, 2.6, 0.5, r'$\text{Adapter}_0$', color_frozen, fontsize=10, fontweight='bold')
ax3.text(-2, 2.0, '(冻结)', fontsize=8, ha='center', va='center', color='#616161', fontstyle='italic')
ax3.text(-2, 1.5, '预训练权重', fontsize=7, ha='center', va='center', color='#9E9E9E')

# 冻结标记
ax3.text(-3.5, 2.2, '[frozen]', fontsize=9, ha='center', va='center', color='#455A64',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#ECEFF1', edgecolor='#90A4AE', alpha=0.8))

# Adapter_1 (可训练)
draw_box(ax3, 2, 2.2, 3.0, 1.8, '', color_trainable, fontsize=10, linewidth=2)
draw_box(ax3, 2, 2.8, 2.6, 0.5, r'$\text{Adapter}_1$', color_trainable, fontsize=10, fontweight='bold')
ax3.text(2, 2.0, '(可训练)', fontsize=8, ha='center', va='center', color='#2E7D32', fontstyle='italic')
ax3.text(2, 1.5, '下游任务适配', fontsize=7, ha='center', va='center', color='#66BB6A')

# 可训练标记
ax3.text(3.5, 2.2, '[train]', fontsize=9, ha='center', va='center', color='#2E7D32',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#66BB6A', alpha=0.8))

# 箭头汇合
draw_arrow(ax3, -2, 3.1, -0.5, 4.1)
draw_arrow(ax3, 2, 3.1, 0.5, 4.1)

# 均值融合
draw_box(ax3, 0, 4.5, 3.2, 0.8, r'均值融合: $\frac{1}{G}\sum_{g=1}^{G}$', '#FFF3E0', fontsize=10, fontweight='bold')

# 输出
draw_arrow(ax3, 0, 4.9, 0, 5.6)
draw_box(ax3, 0, 6.0, 3.0, 0.6, r'融合后表示 $\mathbf{X}_{adapter}$', color_input, fontsize=10)

# 阶段说明
ax3.text(0, 7.2, r'预训练阶段: G=1 (仅 $\mathrm{Adapter}_0$)', fontsize=9, ha='center',
         color='#1565C0', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.8))
ax3.text(0, 8.2, r'下游训练阶段: G=2 ($\mathrm{Adapter}_0$ + $\mathrm{Adapter}_1$)', fontsize=9, ha='center',
         color='#2E7D32', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.8))

# 注入层数说明
ax3.text(0, 9.2, '默认注入 ESM2 最后 $N_{adapter}=16$ 层', fontsize=9, ha='center',
         color='#6A1B9A',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#6A1B9A', alpha=0.8))


plt.tight_layout(pad=2.0)
plt.savefig('/Users/toxic/school/cuiby/cby_latex/figures/adapter_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/Users/toxic/school/cuiby/cby_latex/figures/adapter_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("图片已保存到 figures/adapter_architecture.png 和 figures/adapter_architecture.pdf")
