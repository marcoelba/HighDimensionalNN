import numpy as np
from scipy.linalg import toeplitz


def data_generation(n, k, p, noise_scale = 0.5, beta=None, W=None):
    # set the covariance matrix
    cov_matrix = np.random.rand(k, k) * 0.1
    cov_matrix = cov_matrix @ cov_matrix.T  # positive semi-definite
    cov_matrix += 1 * np.diag(np.ones(k))
    np.linalg.inv(cov_matrix)

    # latent space
    Z = np.random.multivariate_normal(mean=np.zeros(k), cov=cov_matrix, size=n)

    # 2. Create transformation matrix W
    if W is None:
        W = np.random.normal(size=(k, p))

        first_half = range(0, int(p/2))
        second_half = range(int(p/2), p)
        half_k = int(k / 2)

        for k_i in range(0, half_k):
            W[k_i, first_half] = 0.0
        for k_i in range(half_k + 1, k):
            W[k_i, second_half] = 0.0

    # 3. Compute X = ZW + noise
    X = Z @ W + np.random.normal(scale=noise_scale, size=(n, p))

    # add outcome
    if beta is None:
        beta = np.random.choice([-1, 1, 0], size=k)
    y = Z @ beta + np.random.normal(scale=noise_scale, size=n)
    y = y[..., None]

    return y, X, Z, beta


def longitudinal_data_generation(n, k, p, n_timepoints, noise_scale = 0.5, beta=None, W=None, beta_time=None):
    # set the covariance matrix
    cov_matrix = np.random.rand(k, k) * 0.1
    cov_matrix = cov_matrix @ cov_matrix.T  # positive semi-definite
    cov_matrix += 1 * np.diag(np.ones(k))
    np.linalg.inv(cov_matrix)

    # latent space
    Z = np.random.multivariate_normal(mean=np.zeros(k), cov=cov_matrix, size=n)

    # 2. Create transformation matrix W
    if W is None:
        W = np.random.normal(size=(k, p))

        first_half = range(0, int(p/2))
        second_half = range(int(p/2), p)
        half_k = int(k / 2)

        for k_i in range(0, half_k):
            W[k_i, first_half] = 0.0
        for k_i in range(half_k + 1, k):
            W[k_i, second_half] = 0.0

    # 3. Compute X = ZW + noise
    X = Z @ W + np.random.normal(scale=noise_scale, size=(n, p))

    # add outcome
    if beta is None:
        beta = np.random.choice([-1, 1, 0], size=k)
    lin_pred = Z @ beta

    # Generate longitudinal y
    if beta_time is None:
        beta_time = np.random.choice([-1, 1, 0], size=n_timepoints)
    y_time = np.zeros([n, n_timepoints])
    for tt in range(n_timepoints):
        y_time[:, tt] = lin_pred + beta_time[tt]

    y_time = y_time + np.random.normal(scale=noise_scale, size=[n, n_timepoints])

    return y_time, X, Z, beta
