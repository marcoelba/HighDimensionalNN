import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# generate data
p = 5
n = 100
np.random.seed(134)
X_train = np.random.randn(n, p)
y_train = X_train.dot(np.ones(p)) + np.random.randn(n) * 0.1

X_test = np.random.randn(5, p)
y_test = X_test.dot(np.ones(p))

model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
y_hat = model.predict(X_train)
np.sqrt(np.mean((y_train - y_hat)**2))

y_hat = model.predict(X_test)
np.sqrt(np.mean((X_test.dot(np.ones(p)) - y_hat)**2))


alpha = 0.2
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


# ----------------------------------------------------
# With preprocessing
# generate data
p = 5
n = 100
np.random.seed(134)
X_train = np.random.randn(n, p) * 0.4
y_train = X_train.dot(np.ones(p)) + np.random.randn(n) * 0.5
y_train = np.exp(y_train)
plt.hist(y_train)
plt.show()

X_test = np.random.randn(5, p) * 0.4
y_test = np.exp(X_test.dot(np.ones(p)))

alpha = 0.2
K = 10

model_fn = LinearRegression

n_train = len(X_train)
n_test = len(X_test)

# Arrays to store fold predictions and thresholds
kf = KFold(n_splits=K, shuffle=True, random_state=42)
tau = []
L_vals = np.zeros((K, n_test))  # Lower bounds from each fold
U_vals = np.zeros((K, n_test))  # Upper bounds from each fold
K_preds = np.zeros((K, n_test))

for fold_idx, (train_idx, cal_idx) in enumerate(kf.split(X_train)):
    sc = StandardScaler()
    y_log = np.log(y_train[train_idx])
    sc.fit(y_log[..., None])
    y_std = sc.transform(y_log[..., None])[..., 0]
    y_std_val = sc.transform(np.log(y_train[cal_idx])[..., None])[..., 0]
    # Train model on training fold
    model = model_fn()
    model.fit(X_train[train_idx], y_std)
    # Compute residuals on calibration fold
    cal_preds = model.predict(X_train[cal_idx])
    residuals = np.abs(y_std_val - cal_preds)
    # Compute quantile with finite-sample correction
    m = len(residuals)
    q_idx = int(np.ceil((1 - alpha) * (m + 1))) - 1  # 0-indexed
    q_idx = min(q_idx, m - 1)
    tau.append(np.sort(residuals)[q_idx])
    # Predict on test set
    test_preds = model.predict(X_test)
    K_preds[fold_idx, :] = test_preds
    # tr
    l_b = test_preds - tau[fold_idx]
    u_b = test_preds + tau[fold_idx]
    l_b = np.exp(sc.inverse_transform(l_b[..., None])[..., 0])
    u_b = np.exp(sc.inverse_transform(u_b[..., None])[..., 0])
    # Store bounds
    L_vals[fold_idx, :] = l_b
    U_vals[fold_idx, :] = u_b
# Compute final intervals (median across folds)
lower_bounds = np.median(L_vals, axis=0)
upper_bounds = np.median(U_vals, axis=0)
intervals = np.column_stack([lower_bounds, upper_bounds])
mean_prediction = K_preds.mean(axis=0)

y_test_std = sc.transform(np.log(y_test)[..., None])[..., 0]

plt.scatter(y=mean_prediction, x=np.arange(0, n_test, step=1), label="Pred")
plt.scatter(y=y_test_std, x=np.arange(0, n_test, step=1), label="True")
plt.vlines(x=np.arange(0, n_test, step=1), ymax=upper_bounds, ymin=lower_bounds, label="Conformal Int")
plt.legend()
plt.show()

# Transform back intervals to the original scale
exp_lower_bounds = np.exp(sc.inverse_transform(lower_bounds[..., None])[..., 0])
exp_upper_bounds = np.exp(sc.inverse_transform(upper_bounds[..., None])[..., 0])
#
exp_mean_prediction = np.exp(sc.inverse_transform(K_preds.mean(axis=0)[..., None])[..., 0])

plt.scatter(y=exp_mean_prediction, x=np.arange(0, n_test, step=1), label="Pred")
plt.scatter(y=y_test, x=np.arange(0, n_test, step=1), label="True")
plt.vlines(x=np.arange(0, n_test, step=1), ymax=exp_upper_bounds, ymin=exp_lower_bounds, label="Conformal Int")
plt.legend()
plt.show()
