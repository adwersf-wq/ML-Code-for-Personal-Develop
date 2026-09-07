import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
import matplotlib
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

# ==========================================================
# SCI绘图风格设置（宋体最终稳定版）
# ==========================================================

# 全局字体设置
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 保存图片时嵌入字体（非常关键）
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 读取数据
df = pd.read_excel('DRH.xlsx')
X = df.iloc[:, :10]
y = df.iloc[:, 10]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 创建管道
feature_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=200,          # 替换为最佳 n_estimators
        max_depth=None,               # 替换为最佳 max_depth
        min_samples_split=2,        # 替换为最佳 min_samples_split
        min_samples_leaf=1,         # 替换为最佳 min_samples_leaf
        max_features=0.6,        # 替换为最佳 max_features
        bootstrap=True,
        max_samples=None,           # 替换为最佳 max_samples
        random_state=42
    ))
])

model = TransformedTargetRegressor(
    regressor=feature_pipeline,
    transformer=StandardScaler()
)

# # 参数网格（根据随机森林调整）
# param_grid = {
#     'regressor__rf__n_estimators': [200, 400, 600, 800],
#     'regressor__rf__max_depth': [None, 5, 10, 15, 20],
#     'regressor__rf__min_samples_split': [2, 5, 10],
#     'regressor__rf__min_samples_leaf': [1, 3, 5, 7],
#     'regressor__rf__max_features': ['sqrt', 0.6, 0.8, None],
#     'regressor__rf__bootstrap': [True],
#     'regressor__rf__max_samples': [None, 0.8, 0.9]
# }
#
# # 网格搜索
# grid_search = GridSearchCV(model, param_grid, scoring='r2', n_jobs=-1, cv=5)
# grid_search.fit(X_train, y_train)
#
# print(f"最佳参数: {grid_search.best_params_}")
# best_model = grid_search.best_estimator_

# 直接训练模型
model.fit(X_train, y_train)
best_model = model

# ==================== 第一部分：十折交叉验证详细评估 ====================
print("\n" + "=" * 60)
print("十折交叉验证详细评估")
print("=" * 60)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_r2_scores = []
cv_rmse_scores = []
cv_mape_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    fold_model = clone(best_model)
    fold_model.fit(X_train_fold, y_train_fold)
    y_val_pred = fold_model.predict(X_val_fold)

    r2 = r2_score(y_val_fold, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
    mape = mean_absolute_percentage_error(y_val_fold, y_val_pred) * 100

    cv_r2_scores.append(r2)
    cv_rmse_scores.append(rmse)
    cv_mape_scores.append(mape)

    print(f"第{fold}折 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

mean_r2 = np.mean(cv_r2_scores)
mean_rmse = np.mean(cv_rmse_scores)
mean_mape = np.mean(cv_mape_scores)
std_r2 = np.std(cv_r2_scores)
std_rmse = np.std(cv_rmse_scores)
std_mape = np.std(cv_mape_scores)

print("\n" + "-" * 40)
print("十折交叉验证平均性能:")
print(f"平均 R²: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"平均 RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"平均 MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")
print("-" * 40)

# ==================== 第二部分：模型性能评估 ====================
print("\n" + "=" * 60)
print("模型性能评估")
print("=" * 60)

y_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"训练集性能:")
print(f"  R²: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAPE: {train_mape:.2f}%")
print(f"测试集性能:")
print(f"  R²: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAPE: {mape:.2f}%")
print(f"性能差异 (训练集 - 测试集):")
print(f"  R²差异: {train_r2 - r2:.4f}")
print(f"  RMSE差异: {train_rmse - rmse:.4f}")
print(f"  MAPE差异: {train_mape - mape:.2f}%")

gap_r2 = train_r2 - r2

if train_mse < 1e-10 or train_r2 > 0.9999:
    print(f"\n❌ 严重警告：训练集MSE接近0 ({train_mse:.2e}) 或R²接近1 ({train_r2:.4f})，模型在训练集上完美拟合！")
    print(f"   这通常意味着严重过拟合。建议：")
    print(f"   1. 增加 min_samples_split 和 min_samples_leaf")
    print(f"   2. 降低 max_depth")
    print(f"   3. 减少 n_estimators 或增加正则化")
elif gap_r2 > 0.15:
    print(f"\n⚠️  严重警告：训练集和测试集R²差异过大 ({gap_r2:.4f})！")
    print(f"   训练集R²: {train_r2:.4f}, 测试集R²: {r2:.4f}")
    print(f"   建议调整参数网格，使用更强的正则化参数。")
elif gap_r2 > 0.1:
    print(f"\n⚠️  警告：训练集R² ({train_r2:.4f}) 明显高于测试集R² ({r2:.4f})，差异为 {gap_r2:.4f}")
    print(f"   可能存在过拟合，建议考虑：")
    print(f"   - 增大 min_samples_split 和 min_samples_leaf")
    print(f"   - 降低 max_depth")
    print(f"   - 减少 n_estimators")
elif gap_r2 > 0.05:
    print(f"\n⚠️  注意：训练集和测试集性能存在一定差异 (R²差异: {gap_r2:.4f})")
    print(f"   训练集R²: {train_r2:.4f}, 测试集R²: {r2:.4f}")
    print(f"   模型拟合情况可接受，但仍有改进空间。")
else:
    print(f"\n✓ 模型拟合良好：训练集和测试集性能接近 (R²差异: {gap_r2:.4f})")

# ==================== 第三部分：模型验证图（Fig.3样式 - 单图，删除拟合线）====================

print("\n" + "=" * 60)
print("生成模型验证图")
print("=" * 60)

plt.figure(figsize=(8, 8))

plt.scatter(
    y_train,
    y_train_pred,
    alpha=0.6,
    edgecolors='black',
    linewidth=0.8,
    color='blue',
    s=90,
    label='训练集'
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6,
    edgecolors='black',
    linewidth=0.8,
    color='red',
    s=90,
    label='测试集'
)

min_val = min(y.min(), y_train_pred.min(), y_pred.min())
max_val = max(y.max(), y_train_pred.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    'k-',
    linewidth=1.5,
    label='y=x理想拟合线'
)

plt.legend(loc='upper left', fontsize=20)

textstr_train = (
    f'训练集\n'
    f'R2 = {train_r2:.3f}\n'
    f'RMSE = {train_rmse:.3f}\n'
    f'MAPE = {train_mape:.3f}%'
)

textstr_cv = (
    f'测试集\n'
    f'R2 = {mean_r2:.3f}\n'
    f'RMSE = {mean_rmse:.3f}\n'
    f'MAPE = {mean_mape:.3f}%'
)

props = dict(
    boxstyle='round',
    facecolor='wheat',
    alpha=0.5
)

plt.text(
    0.98,
    0.02,
    textstr_train + '\n\n' + textstr_cv,
    transform=plt.gca().transAxes,
    fontsize=20,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=props
)

plt.xlabel('实际值', fontsize=22)
plt.ylabel('预测值', fontsize=22)

plt.title(
    'RF模型验证图（HHV）',
    fontsize=22
)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    'RF_model_validation.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ==================== 第四部分：特征重要性分析（分别绘制）====================
print("\n" + "=" * 60)
print("生成特征重要性分析图...")
print("=" * 60)

# 从最佳模型中提取RF模型和标准化器
rf_model = best_model.regressor_.named_steps['rf']
scaler = best_model.regressor_.named_steps['scaler']

# 获取特征名称（从原始数据列名）
feature_names = X.columns.tolist()
print("特征名称:", feature_names)

# ---------- 图4a：内置特征重要性柱状图 ----------
print("绘制特征重要性柱状图 (Fig.4a)...")
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': feature_importances
})

# 打印特征重要性（从高到低）
print("\n特征重要性排序（从高到低）：")
importance_desc = importance_df.sort_values('重要性', ascending=False)
for idx, row in importance_desc.iterrows():
    print(f"{row['特征']}: {row['重要性']:.6f}")

# 绘图使用升序排列（条形图从低到高）
plot_df = importance_df.sort_values('重要性', ascending=True)

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
bars = plt.barh(plot_df['特征'], plot_df['重要性'], color=colors)
plt.xlabel('重要性得分', fontsize=12)
plt.ylabel('输入特征', fontsize=12)
plt.title('特征重要性分析', fontsize=14, fontweight='bold')

# 在条形上添加数值标签
for bar, val in zip(bars, plot_df['重要性']):
    plt.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
             va='center', fontsize=9)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('RF_feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.show()
print("特征重要性柱状图已保存为 'RF_feature_importance_bar.png'")

# ---------- 图4f：SHAP蜂群图（定制）----------
print("\n绘制SHAP蜂群图 (Fig.4f)...")
X_train_scaled = scaler.transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_scaled)

# 绘制蜂群图，并捕获返回的轴对象
plt.figure(figsize=(10, 6))
ax = shap.summary_plot(shap_values, X_train_scaled, feature_names=feature_names, show=False)

if ax is None:
    ax = plt.gca()

# 强制设置纵轴标签为特征名称
# 设置刻度位置为0~9（共10个特征）
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names)

# 打印纵轴数字与特征名称的对应关系（供手动修改参考）
print("\nSHAP图纵轴数字与特征对应关系（从上到下，0为顶部）：")
for i, name in enumerate(feature_names):
    print(f"{i}: {name}")

# 删除图中的多余线条（竖线、虚线）
lines_to_remove = []
for line in ax.lines:
    # 判断是否为竖线（x坐标恒定）或虚线样式
    if line.get_linestyle() != '-' or (len(line.get_xdata()) == 2 and abs(line.get_xdata()[0] - line.get_xdata()[1]) < 1e-6):
        lines_to_remove.append(line)
for line in lines_to_remove:
    line.remove()

# 修改颜色条
cbar_ax = None
# 查找颜色条轴（通常是有colorbar属性的轴）
for ax_i in plt.gcf().get_axes():
    if hasattr(ax_i, 'colorbar'):
        cbar_ax = ax_i
        break
if cbar_ax is None:
    # 尝试从图像中获取颜色条
    for im in ax.images:
        if im.colorbar is not None:
            cbar_ax = im.colorbar.ax
            break

if cbar_ax is not None:
    # 设置颜色条标签
    cbar_ax.set_ylabel('特征值', fontsize=10)
    # 获取颜色条刻度位置
    ticks = cbar_ax.get_yticks()
    if len(ticks) >= 2:
        # 将首尾两个刻度标签改为'低'和'高'，中间保留原数字
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        tick_labels[0] = '低'
        tick_labels[-1] = '高'
        cbar_ax.set_yticklabels(tick_labels)

ax.set_xlabel('SHAP值 (对模型输出的影响)', fontsize=12)
ax.set_title('RF模型SHAP特征值分布（HHV）', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('RF_shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()
print("SHAP蜂群图已保存为 'RF_shap_beeswarm.png'")

# ==========================================================
# 单变量SHAP依赖图（带趋势曲线）
# ==========================================================
print("\n" + "=" * 60)
print("绘制单变量SHAP依赖图（带趋势曲线）")
print("=" * 60)

# 按重要性排序
sorted_features = importance_desc['特征'].tolist()

# 配色
cmap = plt.get_cmap("RdBu_r")

# 循环绘图
for feature in sorted_features:

    print(f"正在绘制: {feature}")

    plt.figure(figsize=(8, 6))

    # =============================
    # 绘制SHAP散点图
    # =============================
    shap.dependence_plot(
        feature,
        shap_values,
        X_train_scaled,
        feature_names=feature_names,
        interaction_index=None,
        show=False,
        alpha=0.75,
        dot_size=60,
        color='#ff7f0e'
    )

    ax = plt.gca()

    # =============================
    # 获取当前特征数据
    # =============================
    feature_idx = feature_names.index(feature)

    x_data = X_train_scaled.iloc[:, feature_idx].values
    y_data = shap_values[:, feature_idx]

    # =============================
    # LOWESS趋势拟合
    # frac越大越平滑
    # =============================
    lowess_result = lowess(
        y_data,
        x_data,
        frac=0.35
    )

    x_smooth = lowess_result[:, 0]
    y_smooth = lowess_result[:, 1]

    # =============================
    # 绘制趋势曲线
    # =============================
    plt.plot(
        x_smooth,
        y_smooth,
        color='#1f77b4',
        linewidth=1.8,
        linestyle='-',
        label='变量趋势曲线'
    )

    # =============================
    # 坐标轴标签
    # =============================
    ax.set_xlabel(
        f'{feature}（标准化值）',
        fontsize=22
        # fontweight='bold'
    )

    ax.set_ylabel(
        'SHAP值',
        fontsize=22
        # fontweight='bold'
    )

    # =============================
    # 标题
    # =============================
    ax.set_title(
        f'{feature} 的SHAP依赖关系',
        fontsize=24
        # fontweight='bold'
    )

    # =============================
    # 刻度字体
    # =============================
    ax.tick_params(
        axis='both',
        labelsize=18
    )

    # =============================
    # 网格线
    # =============================
    ax.grid(
        True,
        linestyle='--',
        alpha=0.3
    )

    # =============================
    # 图例
    # =============================
    # ax.legend(
    #     fontsize=18,
    #     loc='upper left',
    #     frameon=True
    # )

    # =============================
    # 边框设置
    # =============================
    for spine in ax.spines.values():

        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_linestyle('-')
        spine.set_color('black')

    # =============================
    # 自动布局
    # =============================
    plt.tight_layout()

    # =============================
    # 保存图片
    # =============================
    safe_feature_name = (
        feature
        .replace('/', '_')
    )

    save_name = f'SHAP_dependence_{safe_feature_name}.png'

    plt.savefig(
        save_name,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

    print(f"已保存: {save_name}")

print("\n所有SHAP单变量依赖图绘制完成！")

# ==================== 第六部分：新数据预测（多文件）====================
# print("\n" + "=" * 60)
# print("开始执行多文件预测功能...")
# print("=" * 60)
#
#
# def predict_new_data(input_file_path, output_file_path):
#     new_data = pd.read_excel(input_file_path)
#     if new_data.shape[1] != 10:
#         raise ValueError(f"输入数据应该有10列，但实际有{new_data.shape[1]}列")
#
#     # 强制将列名设置为训练数据的列名（假设列顺序一致）
#     # 如果列顺序不一致，需要先按列名对齐，此处简化处理
#     new_data.columns = X.columns
#
#     predictions = best_model.predict(new_data)
#     result_df = new_data.copy()
#     result_df['HHV'] = predictions
#     result_df.to_excel(output_file_path, index=False)
#     print(f"预测完成！结果已保存到: {output_file_path}")
#     print(f"预测数据形状: {result_df.shape}")
#     return result_df
#
#
# # 定义多个输入输出文件对（请根据实际情况修改文件名）
# file_pairs = [
#     ("new_data_corncob.xlsx", "prediction_results_RF_corncob_HHV.xlsx"),
#     ("new_data_wheat_shell.xlsx", "prediction_results_RF_wheat_shell_HHV.xlsx"),
#     ("new_data_walnut_shell.xlsx", "prediction_results_RF_walnut_shell_HHV.xlsx")
# ]
#
# for input_file, output_file in file_pairs:
#     print(f"\n处理文件: {input_file} -> {output_file}")
#     try:
#         prediction_results = predict_new_data(input_file, output_file)
#         print("预测结果预览:")
#         print(prediction_results.head())
#         print("-" * 40)
#     except FileNotFoundError:
#         print(f"错误: 找不到输入文件 '{input_file}'")
#     except Exception as e:
#         print(f"预测过程中发生错误: {str(e)}")
#         import traceback
#
#         traceback.print_exc()
#
# print("\n" + "=" * 60)
# print("所有文件预测完成！")
# print("=" * 60)