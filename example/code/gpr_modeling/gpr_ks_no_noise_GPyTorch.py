import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import MultitaskMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MultitaskKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
import joblib
import os

# 数据准备和预处理（同上）

filename_suffix = 'no_noise_from_fmu'

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '../data')

X = np.load(os.path.join(data_dir, f'X_{filename_suffix}.npy'))
Y = np.load(os.path.join(data_dir, f'Y_{filename_suffix}.npy'))


# 对数据进行子采样，减少训练样本数量
X_sampled, _, Y_sampled, _ = train_test_split(X, Y, train_size=5000, random_state=42)

# 数据预处理
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X_sampled)
Y_scaled = Y_scaler.fit_transform(Y_sampled)
print("X_scaler mean:", X_scaler.mean_)
print("X_scaler scale:", X_scaler.scale_)
print("Y_scaler mean:", Y_scaler.mean_)
print("Y_scaler scale:", Y_scaler.scale_)


# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)


# 定义模型（同上）

# 初始化似然和模型
num_tasks = Y_train.shape[1]
likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
model = MultitaskGPModel(X_train, Y_train, likelihood, num_tasks=num_tasks)

if torch.cuda.is_available():
    likelihood = likelihood.cuda()
    model = model.cuda()

# 训练模型（同上）

# 模型预测和反标准化
model.eval()
likelihood.eval()

with torch.no_grad():
    preds = model(X_test)
    mean = preds.mean
    Y_pred_scaled = mean.cpu().numpy()
    Y_pred_orig = Y_scaler.inverse_transform(Y_pred_scaled)
    Y_test_orig = Y_scaler.inverse_transform(Y_test.cpu().numpy())

# 保存模型和缩放器（同上）
