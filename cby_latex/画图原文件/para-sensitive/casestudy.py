import numpy as np
import matplotlib.pyplot as plt

# 数据
noise_ratios = ["GHCF", "MBSSL", "POGCN"]

Beibei = [5.4, 6.4, 8.3]
Taobao = [5.9, 7.6, 7.4]



bar_width = 0.35  # 每个柱子的宽度稍微调宽一些，因为只剩下两个
index = np.arange(len(noise_ratios))  # x轴位置

# 创建图形
plt.figure(figsize=(9, 7))

# 绘制两个算法的柱状图
plt.bar(index, Beibei, bar_width, label='Beibei', color='#4c91c2')
plt.bar(index + bar_width, Taobao, bar_width, label='Taobao', color='#d93d3d')

# 添加横向网格线
plt.grid(axis='y', linestyle='-', color='gray', alpha=0.7)

# 添加标签和标题
# plt.xlabel('Dataset', fontsize=24)
plt.ylabel('Recall@10 Increase (%)', fontsize=28)
plt.xticks(index + bar_width / 2, noise_ratios, fontsize=24)  # 设置x轴的刻度位置，使刻度居中

# 设置Y轴每3个数字一个刻度
y_ticks = np.arange(0, 12, 3)  # 从0到15，每3个一跳
plt.yticks(y_ticks, fontsize=24)

# 将图例放在左上角并放大字体
plt.legend(loc='upper left', fontsize=24, title_fontsize=24)

# 图形美化
# plt.title('(a) Beibei')
plt.tight_layout()
plt.savefig('case_study1.pdf', format='pdf', dpi=600)  # 保存矢量图并设置dpi

# 显示图形
plt.show()


