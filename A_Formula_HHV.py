import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.ticker import MaxNLocator

# ===============================
# SCI论文风格字体
# ===============================
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 32
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22


# ===============================
# 读取数据
# ===============================
data = pd.read_excel("DRH.xlsx")

X_cols = ['Ash (%)', 'FC (%)', 'C (%)', 'H (%)', 'O (%)']
y_col = 'HHV (MJ/kg)'

X = data[X_cols]
y = data[y_col]


# ===============================
# Pearson相关性
# ===============================
print("\n===== Pearson相关性 =====")

for col in X_cols:

    r = np.corrcoef(X[col], y)[0,1]

    relation = "正相关" if r > 0 else "负相关"

    print(f"{col} : {relation} (r = {r:.3f})")


# ===============================
# 构建二次项
# ===============================
X_poly = X.copy()

X_poly['Ash2'] = X['Ash (%)']**2
X_poly['FC2'] = X['FC (%)']**2
X_poly['C2'] = X['C (%)']**2


# ===============================
# 回归模型
# ===============================
model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)


# ===============================
# 输出经验公式
# ===============================
coef = model.coef_
intercept = model.intercept_

print("\n===== 拟合经验公式 =====")

formula = f"HHV = {intercept:.4f}"

names = list(X_poly.columns)

for i in range(len(names)):

    if coef[i] >= 0:
        formula += f" + {coef[i]:.4f}*{names[i]}"
    else:
        formula += f" - {abs(coef[i]):.4f}*{names[i]}"

print(formula)


# ===============================
# 模型评价指标
# ===============================
R2 = r2_score(y, y_pred)
RMSE = np.sqrt(mean_squared_error(y, y_pred))
MAPE = np.mean(np.abs((y - y_pred)/y))*100

print("\n===== 总体模型性能 =====")

print("R2 =", R2)
print("RMSE =", RMSE)
print("MAPE =", MAPE)


# ==================================================
# 图1 真实值 vs 预测值折线图（按真实值排序）
# ==================================================

# 按真实值排序
order = np.argsort(y)

y_sorted = y.iloc[order]
y_pred_sorted = y_pred[order]

index = np.arange(len(y_sorted))

plt.figure(figsize=(8,5))

# SCI配色
exp_color = "#1f77b4"
pred_color = "#ff7f0e"

plt.plot(index, y_sorted,
         color=exp_color,
         marker='o',
         linewidth=1.8,
         label="真实值")

plt.plot(index, y_pred_sorted,
         color=pred_color,
         marker='o',
         linestyle='None',
         linewidth=1.8,
         label="预测值")

plt.xlabel("排序后的样本序号")
plt.ylabel("HHV (MJ/kg)")

plt.title("真实值与拟合公式预测值对比")

plt.legend()

plt.tight_layout()

plt.savefig("HHV_validation_line.png", dpi=300)

plt.show()


# ==================================================
# 图2 单变量响应曲线
# ==================================================

means = X.mean()

for col in X_cols:

    x_range = np.linspace(X[col].min(), X[col].max(), 100)

    df = pd.DataFrame({
        'Ash (%)': means['Ash (%)'],
        'FC (%)': means['FC (%)'],
        'C (%)': means['C (%)'],
        'H (%)': means['H (%)'],
        'O (%)': means['O (%)']
    }, index=range(100))

    df[col] = x_range

    df['Ash2'] = df['Ash (%)']**2
    df['FC2'] = df['FC (%)']**2
    df['C2'] = df['C (%)']**2

    y_curve = model.predict(df)

    plt.figure(figsize=(8,6.45))

    plt.plot(x_range, y_curve,
             color="#1f77b4",
             linewidth=2)

    plt.xlabel(col, fontsize=26)
    plt.ylabel("HHV(MJ/kg)", fontsize=26)

    plt.title(f"{col} 对HHV的响应", fontsize=28)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # ---------- 添加边框设置 ----------
    ax = plt.gca()  # 获取当前坐标轴
    for spine in ax.spines.values():
        spine.set_visible(True)  # 确保边框可见（默认左右上右下四条）
        spine.set_linewidth(1.0)  # 边框线宽
        spine.set_linestyle('-')  # 实线
        spine.set_color('black')  # 黑色

    # ---------- y轴整数刻度 ----------
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    safe_name = col.replace("(","").replace(")","").replace("°","").replace("/","_")

    plt.tight_layout()

    plt.savefig(f"{safe_name}_response.png", dpi=300)

    plt.show()