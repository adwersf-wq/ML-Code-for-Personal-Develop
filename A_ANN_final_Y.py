import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
import matplotlib


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
df = pd.read_excel('DR.xlsx')
X = df.iloc[:, :10]
y = df.iloc[:, 10]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 创建包含自动归一化的管道
feature_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ann', MLPRegressor(
        # hidden_layer_sizes=(150,100),  # 替换为最佳 hidden_layer_sizes
        # activation='relu',  # 替换为最佳 activation
        # solver='adam',  # 替换为最佳 solver
        # alpha=0.01,  # 替换为最佳 alpha
        # learning_rate_init=0.01,  # 替换为最佳 learning_rate_init

        hidden_layer_sizes=(150,250),  # 替换为最佳 hidden_layer_sizes
        activation='relu',  # 替换为最佳 activation
        solver='adam',  # 替换为最佳 solver
        alpha=0.1,  # 替换为最佳 alpha
        learning_rate_init=0.005,  # 替换为最佳 learning_rate_init
        # random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=1000
    ))
])

# 对目标变量也进行归一化
model = TransformedTargetRegressor(
    regressor=feature_pipeline,
    transformer=StandardScaler()
)

# # 定义参数网格（GridSearchCV使用）
# param_grid = {
#     'regressor__ann__hidden_layer_sizes': [(50,), (100,), (150,), (200,), (250,), (300,), (50, 50), (100, 50), (150, 50), (200, 50), (250, 50), (300, 50), (50, 100), (100, 100), (150, 100), (200, 100), (250, 100), (300, 100), (50, 150), (100, 150), (150, 150), (200, 150), (250, 150), (300, 150), (50, 200), (100, 200), (150, 200), (200, 200), (250, 200), (300, 200), (50, 250), (100, 250), (150, 250), (200, 250), (250, 250), (300, 250), (50, 300), (100, 300), (150, 300), (200, 300), (250, 300), (300,300)],
#     'regressor__ann__activation': ['relu', 'tanh', 'logistic'],
#     'regressor__ann__solver': ['adam', 'sgd'],
#     'regressor__ann__alpha': [0.0001, 0.001, 0.01, 0.1],          # 均匀分布转换为列表
#     'regressor__ann__learning_rate_init': [0.001, 0.005, 0.01]    # 均匀分布转换为列表
# }
#
# # 网格搜索
# grid_search = GridSearchCV(
#     model,
#     param_grid=param_grid,
#     scoring='r2',
#     n_jobs=-1,
#     cv=5,
#     verbose=1
# )
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

    # 克隆并训练模型（管道会自动处理标准化）
    fold_model = clone(best_model)
    fold_model.fit(X_train_fold, y_train_fold)

    # 预测验证集
    y_val_pred = fold_model.predict(X_val_fold)

    # 计算指标
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

# 过拟合诊断
gap_r2 = train_r2 - r2

if train_mse < 1e-10 or train_r2 > 0.9999:
    print(f"\n❌ 严重警告：训练集MSE接近0 ({train_mse:.2e}) 或R²接近1 ({train_r2:.4f})，模型在训练集上完美拟合！")
    print(f"   这通常意味着严重过拟合。建议：")
    print(f"   1. 增加 alpha（正则化强度）")
    print(f"   2. 减小网络规模 hidden_layer_sizes")
    print(f"   3. 增加 early_stopping 的耐心或降低 max_iter")
elif gap_r2 > 0.15:
    print(f"\n⚠️  严重警告：训练集和测试集R²差异过大 ({gap_r2:.4f})！")
    print(f"   训练集R²: {train_r2:.4f}, 测试集R²: {r2:.4f}")
    print(f"   建议调整参数网格，使用更强的正则化参数。")
elif gap_r2 > 0.1:
    print(f"\n⚠️  警告：训练集R² ({train_r2:.4f}) 明显高于测试集R² ({r2:.4f})，差异为 {gap_r2:.4f}")
    print(f"   可能存在过拟合，建议考虑：")
    print(f"   - 增加 alpha")
    print(f"   - 减小网络规模")
    print(f"   - 增加 validation_fraction 或 early_stopping")
elif gap_r2 > 0.05:
    print(f"\n⚠️  注意：训练集和测试集性能存在一定差异 (R²差异: {gap_r2:.4f})")
    print(f"   训练集R²: {train_r2:.4f}, 测试集R²: {r2:.4f}")
    print(f"   模型拟合情况可接受，但仍有改进空间。")
else:
    print(f"\n✓ 模型拟合良好：训练集和测试集性能接近 (R²差异: {gap_r2:.4f})")

# ==================== 第三部分：模型验证图（Fig.3样式 - 单图，无拟合线）====================
print("\n" + "=" * 60)
print("生成模型验证图 (Fig.3 样式)...")
print("=" * 60)

plt.figure(figsize=(8, 8))

# 散点：训练集蓝色，测试集红色
plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolors='black', linewidth=0.8, color='blue', s=90, label='训练集')
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.8, color='red', s=90, label='测试集')

# 对角线（理想拟合线）
min_val = min(y.min(), y_train_pred.min(), y_pred.min())
max_val = max(y.max(), y_train_pred.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5, label='y=x理想拟合线')

# 图例放在左上角，字号10
plt.legend(loc='upper left', fontsize=20)

# 右下角性能标注（两行，字号10，使用普通 R2 避免字体问题）
textstr_train = f'训练集\nR2 = {train_r2:.3f}\nRMSE = {train_rmse:.3f}\nMAPE = {train_mape:.3f}%'
textstr_cv = f'测试集\nR2 = {mean_r2:.3f}\nRMSE = {mean_rmse:.3f}\nMAPE = {mean_mape:.3f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.98, 0.02, textstr_train + '\n\n' + textstr_cv, transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.xlabel('实际值', fontsize=22)
plt.ylabel('预测值', fontsize=22)
plt.title('ANN模型验证图（产率）', fontsize=22)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ANN_model_validation.png', dpi=300, bbox_inches='tight')
plt.show()
print("模型验证图已保存为 'ANN_model_validation.png'")

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
#     new_data.columns = X.columns
#
#     predictions = best_model.predict(new_data)
#     result_df = new_data.copy()
#     result_df['产量'] = predictions
#     result_df.to_excel(output_file_path, index=False)
#     print(f"预测完成！结果已保存到: {output_file_path}")
#     print(f"预测数据形状: {result_df.shape}")
#     return result_df
#
#
# # 定义多个输入输出文件对（请根据实际情况修改文件名）
# file_pairs = [
#     ("new_data_corncob.xlsx", "prediction_results_ANN_corncob_Yield.xlsx"),
#     ("new_data_wheat_shell.xlsx", "prediction_results_ANN_wheat_shell_Yield.xlsx"),
#     ("new_data_walnut_shell.xlsx", "prediction_results_ANN_walnut_shell_Yield.xlsx")
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
#         traceback.print_exc()
#
# print("\n" + "=" * 60)
# print("所有文件预测完成！")
# print("=" * 60)