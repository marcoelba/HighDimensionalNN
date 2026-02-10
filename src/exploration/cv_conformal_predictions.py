import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# generate data
p = 5
n = 100
np.random.seed(134)
X_train = np.random.randn(n, p)
y_train = X_train.dot(np.ones(p)) + np.random.randn(n)*0.5
X_test = np.random.randn(5, p)
y_test = X_test.dot(np.ones(p))

model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
y_hat = model.predict(X_train)
np.sqrt(np.mean((y_train - y_hat)**2))

y_hat = model.predict(X_test)
np.sqrt(np.mean((X_test.dot(np.ones(p)) - y_hat)**2))


alpha = 0.05
K = 10

model_fn = LinearRegression

n_train = len(X_train)
n_test = len(X_test)

# Arrays to store fold predictions and thresholds
L_vals = np.zeros((K, n_test))  # Lower bounds from each fold
U_vals = np.zeros((K, n_test))  # Upper bounds from each fold
K_preds = np.zeros((K, n_test))

kf = KFold(n_splits=K, shuffle=True, random_state=42)

for fold_idx, (train_idx, cal_idx) in enumerate(kf.split(X_train)):
    # Train model on training fold
    model = model_fn()
    model.fit(X_train[train_idx], y_train[train_idx])
    # Compute residuals on calibration fold
    cal_preds = model.predict(X_train[cal_idx])
    residuals = np.abs(y_train[cal_idx] - cal_preds)
    # Compute quantile with finite-sample correction
    m = len(residuals)
    q_idx = int(np.ceil((1 - alpha) * (m + 1))) - 1  # 0-indexed
    q_idx = min(q_idx, m - 1)
    tau = np.sort(residuals)[q_idx]
    # Predict on test set
    test_preds = model.predict(X_test)
    K_preds[fold_idx, :] = test_preds
    # Store bounds
    L_vals[fold_idx, :] = test_preds - tau
    U_vals[fold_idx, :] = test_preds + tau

# Compute final intervals (median across folds)
lower_bounds = np.median(L_vals, axis=0)
upper_bounds = np.median(U_vals, axis=0)
intervals = np.column_stack([lower_bounds, upper_bounds])
mean_prediction = K_preds.mean(axis=0)

plt.scatter(y=mean_prediction, x=np.arange(0, n_test, step=1), label="Pred")
plt.scatter(y=y_test, x=np.arange(0, n_test, step=1), label="True")
plt.vlines(x=np.arange(0, n_test, step=1), ymax=upper_bounds, ymin=lower_bounds, label="Conformal Int")
plt.legend()
plt.show()
