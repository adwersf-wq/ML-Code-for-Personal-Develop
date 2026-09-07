import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 字体设置（中英文同时支持）
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 读取 Excel
file_path = 'DRH.xlsx'
df = pd.read_excel(file_path)

# Pearson相关矩阵
correlation_matrix = df.corr(method='pearson')

# 创建画布
plt.figure(figsize=(13, 13))

# 绘制热力图
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    cbar=True,
    square=True,                 # 保证是正方形
    annot_kws={"size": 20},
    xticklabels=correlation_matrix.columns,
    yticklabels=correlation_matrix.index
)

# ===== 调整右侧colorbar数字大小 =====
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

# 坐标轴字体
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

# 标题
plt.title('生物质热解炭HHV的Pearson相关系数图', fontsize=27)

# 调整布局（防止裁剪）
plt.tight_layout()

# 保存图片
plt.savefig(
    'Pearson_correlation.png',
    dpi=600,           # 论文级清晰度
    format='png'
)

# 显示
plt.show()