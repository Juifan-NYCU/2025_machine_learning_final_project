import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 設定隨機種子確保可重現
torch.manual_seed(42)
np.random.seed(42)

# 真實函數
def true_sin(x):
    return np.sin(x)

# 生成訓練數據 (在 x 上加噪聲)
n_train = 500
x_clean = np.linspace(-np.pi, np.pi, n_train)
y_clean = true_sin(x_clean)

# 輸入加噪聲 (標準差可調整，這裡用 0.3)
input_noise_std = 0.3
x_train_noisy = x_clean + np.random.normal(0, input_noise_std, size=x_clean.shape)
x_train_noisy = torch.FloatTensor(x_train_noisy).unsqueeze(1)  # (500, 1)
y_train = torch.FloatTensor(y_clean).unsqueeze(1)              # (500, 1)

# 驗證集 (不加噪聲，用來看真實泛化能力)
n_val = 1000
x_val_clean = np.linspace(-np.pi, np.pi, n_val)
y_val_clean = true_sin(x_val_clean)
x_val = torch.FloatTensor(x_val_clean).unsqueeze(1)
y_val = torch.FloatTensor(y_val_clean).unsqueeze(1)

# 神經網絡模型 (2個隱藏層，每層50個神經元，ReLU激活)
class SinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SinNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練
n_epochs = 8000
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    pred_train = model(x_train_noisy)
    loss_train = criterion(pred_train, y_train)
    loss_train.backward()
    optimizer.step()

    # 驗證 (用乾淨的 x)
    model.eval()
    with torch.no_grad():
        pred_val = model(x_val)
        loss_val = criterion(pred_val, y_val)

    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())

    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {loss_train.item():.6f}, Val Loss: {loss_val.item():.6f}")

# 最終預測 (用乾淨的測試點)
model.eval()
with torch.no_grad():
    x_test = torch.FloatTensor(np.linspace(-np.pi, np.pi, 1000).reshape(-1,1))
    y_pred = model(x_test).numpy().flatten()
    y_true = true_sin(x_test.numpy().flatten())

# 計算錯誤
mse = np.mean((y_pred - y_true)**2)
max_error = np.max(np.abs(y_pred - y_true))
print(f"\nFinal MSE: {mse:.6f}")
print(f"Final Max Absolute Error: {max_error:.6f}")

# 繪圖
plt.figure(figsize=(15, 10))

# 1. 真實函數 vs 預測
plt.subplot(2, 2, 1)
plt.plot(x_test, y_true, 'b-', linewidth=2, label='True sin(x)')
plt.plot(x_test, y_pred, 'r--', linewidth=2, label='NN Prediction (with input noise training)')
plt.scatter(x_clean, y_clean, c='gray', s=10, alpha=0.6, label='Training points (clean)')
plt.title('True Function vs Neural Network Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 2. 訓練點 (帶噪聲輸入)
plt.subplot(2, 2, 2)
plt.scatter(x_train_noisy.numpy(), y_train.numpy(), c='orange', s=10, alpha=0.7, label='Noisy input training points')
plt.plot(x_test, y_true, 'b-', linewidth=2, label='True sin(x)')
plt.title('Training Data (Input with Noise σ=0.3)')
plt.xlabel('Noisy x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)

# 3. Loss curves
plt.subplot(2, 2, 3)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.yscale('log')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# 4. 殘差圖
plt.subplot(2, 2, 4)
residual = y_pred - y_true
plt.plot(x_test, residual, 'g-')
plt.axhline(0, color='black', linestyle='--')
plt.title(f'Residual (Max error = {max_error:.4f})')
plt.xlabel('x')
plt.ylabel('Prediction - True')
plt.grid(True)

plt.tight_layout()
plt.show()