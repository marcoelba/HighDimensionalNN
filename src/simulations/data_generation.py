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
        y_time[:, tt] = lin_pred[:, tt] + beta_time[tt]

    y_time = y_time + np.random.normal(scale=noise_scale, size=[n, n_timepoints])

    return y_time, X, Z, beta


def multi_longitudinal_data_generation(
    n, k, p,
    n_timepoints, n_measurements,
    p_static=0,
    p_interventions=0,
    noise_scale = 0.5,
    W=None,
    beta=None, beta_time=None, beta_static=None, beta_interventions=None,
    missing_prob=0.0
):
    """
    Generate synthetic multi-dimensional longitudinal data with latent structure.
    
    This function creates synthetic longitudinal data where observations are influenced
    by latent factors, time effects, static covariates, and intervention covariates.
    
    Parameters
    ----------
    n : int
        Number of subjects/patients in the dataset.
    k : int
        Number of latent factors/dimensions.
    p : int
        Number of observed features per measurement.
    n_timepoints : int
        Number of time points in the longitudinal data.
    n_measurements : int
        Number of measurements per subject.
    p_static : int, optional
        Number of static patient-level covariates (e.g., clinical anthropometrics).
        Default is 0 (no static covariates).
    p_interventions : int, optional
        Number of intervention/measurement-level covariates.
        Default is 0 (no intervention covariates).
    noise_scale : float, optional
        Standard deviation of Gaussian noise added to observations.
        Default is 0.5.
    beta : array-like, optional
        Coefficient matrix of shape (k, n_timepoints) for latent factor effects.
        If None, randomly initialized with values in {-1, 0, 1}.
    W : array-like, optional
        Transformation matrix of shape (k, p) mapping latent space to observed features.
        If None, creates a structured matrix where first half of latent factors
        affect first half of features and vice versa.
    beta_time : array-like, optional
        Time-specific effects of shape (n_timepoints,).
        If None, randomly initialized with values in {-1, 0, 1}.
    beta_static : array-like, optional
        Coefficients for static covariates of shape (p_static,).
        If None, randomly initialized with values in {-1, 1}.
        Only used if p_static > 0.
    beta_interventions : array-like, optional
        Coefficients for intervention covariates of shape (n_measurements, p_interventions).
        If None, randomly initialized with values in {-1, 1}.
        Only used if p_interventions > 0.
    missing_prob : float = 0
        If > 0 introduces random missing values in the arrays for whole measurements
    Returns
    -------
    y_time : ndarray
        Longitudinal response/target variable of shape (n, n_measurements, n_timepoints).
        Contains the generated outcome data with noise.
    X : ndarray
        Observed features of shape (n, n_measurements, p).
        Generated from latent factors through transformation matrix W.
    Z : ndarray
        Latent factors of shape (n, k).
        Multivariate normal samples used to generate the data.
    beta : ndarray
        Coefficient matrix of shape (k, n_timepoints) used for generating the response.
    
    Notes
    -----
    The data generation process follows:
    1. Generate latent factors Z ~ MVN(0, Σ) where Σ is a random covariance matrix
    2. Create observed features X = Z @ W + noise
    3. Generate linear predictor: lin_pred = Z @ beta + static_effects + intervention_effects
    4. Add time effects: y_time = lin_pred + beta_time + noise
    
    The transformation matrix W is structured such that:
    - First half of latent factors only affect first half of features
    - Second half of latent factors only affect second half of features
    
    Examples
    --------
    >>> y, X, Z, beta = multi_longitudinal_data_generation(
    ...     n=100, k=3, p=10, n_timepoints=5, n_measurements=4,
    ...     p_static=2, p_interventions=1, noise_scale=0.1
    ... )
    >>> y.shape
    (100, 4, 5)
    >>> X.shape
    (100, 4, 10)
    >>> Z.shape
    (100, 3)
    >>> beta.shape
    (3, 5)
    """
    out_dict = dict()

    # set the covariance matrix
    cov_matrix = np.random.rand(k, k) * 0.1
    cov_matrix = cov_matrix @ cov_matrix.T  # positive semi-definite
    cov_matrix += 1 * np.diag(np.ones(k))
    np.linalg.inv(cov_matrix)

    # latent space
    Z = np.random.multivariate_normal(
        mean=np.zeros(k),
        cov=cov_matrix,
        size=n
    )
    out_dict["Z"] = Z
    
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

    if beta is None:
        beta = np.random.choice([-1, 1, 0], size=(k, n_timepoints))
    
    # Time effect
    if beta_time is None:
        # same effect for all measurements
        beta_time = np.random.choice([-1, 1, 0], size=n_timepoints)

    # Static patient level features, like clinical anthropometrics
    if p_static > 0:
        X_static = np.random.normal(scale=1., size=(n, p_static))
        # expand over M dimension
        X_static = np.repeat(np.expand_dims(X_static, axis=1), n_measurements, axis=1)
        if beta_static is None:
            beta_static = np.random.choice([-1, 1], size=p_static)
        out_dict["X_static"] = X_static
    
    # Measurements/Internventions level covariates
    if p_interventions > 0:
        X_interventions = np.random.normal(scale=1., size=(n, p_interventions))
        if beta_interventions is None:
            beta_interventions = np.random.choice([-1, 1], size=(n_measurements, p_interventions))
        out_dict["X_interventions"] = X_interventions

    # 3. Compute X = ZW + noise
    X = np.zeros([n, n_measurements, p])
    for m in range(n_measurements):
        X[:, m, :] = Z @ W + np.random.normal(scale=noise_scale, size=(n, p))
    out_dict["X"] = X

    lin_pred = np.zeros([n, n_measurements, n_timepoints])
    for m in range(n_measurements):
        lin_pred[:, m, :] = Z @ beta
    
    # Add static patient effects if present
    if p_static > 0:
        for m in range(n_measurements):
            for t in range(n_timepoints):
                lin_pred[:, m, t] = lin_pred[:, m, t] + X_static[:, m, :] @ beta_static

    # Add static patient effects if present
    if p_interventions > 0:
        for m in range(n_measurements):
            for t in range(n_timepoints):
                lin_pred[:, m, t] = lin_pred[:, m, t] + X_interventions @ beta_interventions[m, :]
    
    # Generate longitudinal y    
    y_time = np.zeros([n, n_measurements, n_timepoints])
    for m in range(n_measurements):
        for tt in range(n_timepoints):
            y_time[:, m, tt] = lin_pred[:, m, tt] + beta_time[tt]

    y_time = y_time + np.random.normal(scale=noise_scale, size=y_time.shape)
    out_dict["y"] = y_time

    # add missing values (nan) if required
    if missing_prob > 0:
        mask = np.random.binomial(1, 1 - missing_prob, size=(n, n_measurements)) # n x M
        # apply to X
        for m in range(n_measurements):
            out_dict["X"][mask[:, m] == 0, m, :] = np.nan
            out_dict["X_static"][mask[:, m] == 0, m, :] = np.nan
            out_dict["y"][mask[:, m] == 0, m, :] = np.nan

    return out_dict
