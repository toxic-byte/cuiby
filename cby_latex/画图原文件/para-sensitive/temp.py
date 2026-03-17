import matplotlib.pyplot as plt
import numpy as np

# 数据集
layers = [0.1, 0.2, 0.3, 0.4, 0.5]  # x轴
beibei = [0.1897, 0.1976, 0.1985, 0.2111, 0.1956]  # Beibei 数据
tmall = [0.1132, 0.1207, 0.0911, 0.0906, 0.0879]  # Tmall 数据
taobao = [0.1091, 0.1102, 0.1043, 0.0904, 0.0891]  # Taobao 数据

# 创建图表并调整大小
plt.figure(figsize=(9, 7))  # 调整figsize来增加图片尺寸

# 绘制折线并标记数据点
plt.plot(layers, beibei, marker='o', label='Beibei', linewidth=3, markersize=10)
plt.plot(layers, tmall, marker='*', label='Tmall', linewidth=3, markersize=10)
plt.plot(layers, taobao, marker='s', label='Taobao', linewidth=3, markersize=10)

# 设置坐标轴标签
plt.xlabel(r'$\tau$', fontsize=30, fontweight='bold', labelpad=-5.0)
plt.ylabel('Recall@10', fontsize=28, fontweight='bold', labelpad=5.0)
plt.subplots_adjust(left=0.1750)  # 增加左侧边距

# 添加图例
plt.legend(loc='best', prop={'weight': 'bold', 'size':24})

# 设置坐标轴范围
plt.ylim(0.08, 0.2101)
plt.xlim(0.0, 0.55)  # 设置x轴范围以适应数据

# 设置 x 轴刻度
plt.xticks(np.arange(0.0, 0.55, 0.1), fontsize=24, fontweight='bold')  # 每个横坐标加1保留

# 设置 y 轴刻度
plt.yticks(np.arange(0.08, 0.22, 0.02), fontsize=24, fontweight='bold')  # 每个纵坐标加10保留

# 显示网格
plt.grid(True)

# 保存为高分辨率的矢量图或位图图像
plt.savefig('temp.pdf', format='pdf', dpi=600)  # 保存矢量图并设置dpi

# 显示图表
plt.show()


