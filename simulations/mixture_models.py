"""
Classical mixture models for Clustering 

- Gaussian mixture with covariance (with random centers or centers on a simplex
- Gamma mixture with rate formulation (proposed in Neufeld, Data thinning paper)


"""
import numpy as np


def gaussian_mixture(K, n, r, Cov_k = 1.0, alpha=1, method='simplex', random_state=422, diag=False):
    rng = np.random.RandomState(random_state)
    # partially defined Covs, default is isotropic 
    if isinstance(Cov_k, float):
        Cov_k = [Cov_k * np.eye(r) for k in range(K)]
    elif len(np.array(Cov_k).shape)== 2:
        Cov_k = [Cov_k for k in range(K)]

    if method=='simplex':
        cluster_centers = simplex_centers(K, r, alpha)
    elif method =='random':
        cluster_centers = random_centers(K, r, alpha, rng)
 
    X, labels = [], []
    for k, center in enumerate(cluster_centers):
        samples = rng.multivariate_normal(mean=center, cov=Cov_k[k], size=n)
        X.append(samples)
        labels.append(np.full(n, k, dtype=int))
    X = np.vstack(X) # (N, r), N = K*n
    labels = np.concatenate(labels) # (N, )
    if diag:
        return X, labels, cluster_centers
    else:
        return X, labels

def find_vertices(p):
    """
    Generate the (p+1) vertices of a p-dimensional equilateral simplex
    with all edge-lengths = 2.
    """
    m = p+1 
    V = np.zeros((m, p))
    V[0,0] = -1
    V[1,0] = 1

    for k in range(2, m):
        mean_k = V[:k, :k-1].sum(axis=0) / k
        mean_k[0] = 1
        V[k,:k-1] = mean_k
        dd = np.dot(mean_k, mean_k)
        V[k,k-1] = np.sqrt(4 - dd)
    return V
    

def simplex_centers(K, r, alpha):
    """
    Generate K centroids in r dimensions using simplex + shifting logic.
    Used by both cluster_gen() and low_rank_gmm().
    """
    base = find_vertices(r) # find all vertices in simplex, will be (r + 1, r)
    cluster_centers = np.zeros((K, r), dtype=float)
    # if K <= r+1
    n0 = min(K, r + 1)
    cluster_centers[:n0, :] = base[:n0, :]
    if K > r + 1:
        e1 = np.zeros(r, dtype=float)
        e1[0] = 1.0
        # How many *full* blocks of d_0 points beyond the first r+1?
        full_blocks, remainder = (K - (r + 1)) // r, (K - (r + 1)) % r
        idx = r + 1
        # For each full block j = 1 … full_blocks, shift by 2 * j * e1
        for j in range(1, full_blocks + 1):
            for i in range(1, r + 1):
                cluster_centers[idx, :] = base[i, :] + 2 * j * e1
                idx += 1
        # Finally the leftover remainder (shift by 2*(full_blocks+1))
        if remainder > 0:
            shift = 2 * (full_blocks + 1)
            for i in range(1, remainder + 1):
                cluster_centers[idx, :] = base[i, :] + shift * e1
                idx += 1
    return alpha * cluster_centers # (K, r)

def random_centers(K, r, alpha, rng):
    cluster_centers = rng.choice([-alpha, 0.0, +alpha], size=(K, r))
    return cluster_centers

def make_gamma_scales_sparse_rs(rng: np.random.RandomState,
    K: int,
    d: int,
    n_signal: int = 25,
    base_log_scale: float = 0.0,
    signal_shift: float = 0.8,
    feature_sigma: float = 0.3,
    cluster_sigma: float = 0.2,
    jitter_sigma: float = 0.05) -> np.ndarray:
    """
    RandomState-based version of sparse scale generator.
    Returns scale matrix (K, d), strictly positive.
    """
    n_signal = min(n_signal, d)

    feat = rng.normal(loc=base_log_scale, scale=feature_sigma, size=d)   # (d,)
    scale = np.exp(feat)[None, :].repeat(K, axis=0)                      # (K,d)

    clus = rng.normal(loc=0.0, scale=cluster_sigma, size=(K, 1))         # (K,1)

    pattern = rng.choice([-1.0, +1.0], size=(K, n_signal))
    pattern = pattern - pattern.mean(axis=0, keepdims=True)

    logS_signal = (
        feat[:n_signal][None, :]
        + clus
        + signal_shift * pattern
        + rng.normal(0.0, jitter_sigma, size=(K, n_signal))
    )
    scale[:, :n_signal] = np.exp(logS_signal)
    return scale


def gamma_mixture(
    n_samples: int = 100,
    n_features: int = 10,
    n_clusters: int = 5,
    prop=None,
    shared_shape: float = 2.0,
    scale=None,
    random_state: int = 42,
    # ---- optional params to control scale generation when scale is None ----
    n_signal: int = 25,
    base_log_scale: float = 0.0,
    signal_shift: float = 0.8,
    feature_sigma: float = 0.3,
    cluster_sigma: float = 0.2,
    jitter_sigma: float = 0.05,
):
    """
    Gamma mixture with cluster-specific scale parameters.

    Output: X, labels, shape, scale
      - X: (n_samples, n_features)
      - labels: (n_samples,) in {0,...,K-1}
      - shape: (n_features,) (shared across clusters here)
      - scale: (K, n_features) cluster-specific scales
    """
    rng = np.random.RandomState(random_state)
    n, K, d = n_samples, n_clusters, n_features

    if prop is None:
        prop = np.ones(K) / K
    else:
        prop = np.asarray(prop, dtype=float)
        if prop.shape != (K,):
            raise ValueError(f"prop must have shape ({K},), got {prop.shape}")
        if np.any(prop < 0):
            raise ValueError("prop must be nonnegative")
        s = prop.sum()
        if s <= 0:
            raise ValueError("prop must sum to > 0")
        prop = prop / s

    labels = rng.choice(K, size=n, p=prop)

    # Generate or validate scale matrix
    if scale is None:
        scale = make_gamma_scales_sparse_rs(
            rng=rng,
            K=K,
            d=d,
            n_signal=n_signal,
            base_log_scale=base_log_scale,
            signal_shift=signal_shift,
            feature_sigma=feature_sigma,
            cluster_sigma=cluster_sigma,
            jitter_sigma=jitter_sigma,
        )
    else:
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (K, d):
            raise ValueError(f"scale must be shape (K, d)=({K},{d}), got {scale.shape}")
        if np.any(scale <= 0):
            raise ValueError("scale must be strictly positive")

    shape = np.full(d, float(shared_shape))

    # Sample
    X = np.zeros((n, d), dtype=float)
    for k in range(K):
        mask = (labels == k)
        n_k = int(mask.sum())
        if n_k > 0:
            X[mask] = rng.gamma(shape, scale[k], size=(n_k, d))

    return X, labels, shape, scale


# Sparse GMM with AR(1) covariance

def ar1_gaussian(n, m, rho=0.5, rng=None):
    """
    Draw n samples from N(0, Sigma) where Sigma_{ij} = rho^{|i-j|}.
    Efficient O(n*m) generation via AR(1) recursion.
    """
    if rng is None:
        rng = np.random.default_rng()

    eps = rng.standard_normal((n, m))
    X = np.empty((n, m), dtype=float)

    # stationary variance is 1 when innovation std = sqrt(1-rho^2)
    X[:, 0] = eps[:, 0]
    s = np.sqrt(1.0 - rho**2)
    for j in range(1, m):
        X[:, j] = rho * X[:, j - 1] + s * eps[:, j]
    return X


def make_signal_means(K=4, p_signal=25, snr=4.0):
    """
    Build mu_k in R^{p_signal} for k=1..K so that ||mu_k||_2 = snr.
    Uses the common 4-pattern construction:
      m1: + + + ... (25)
      m2: + (13), - (12)
      m3: - (13), + (12)
      m4: - - - ... (25)
    """
    if K != 4:
        raise ValueError("This mean pattern is defined for K=4 per the described setup.")

    if p_signal != 25:
        raise ValueError("This mean pattern assumes 25 signal features (13 + 12 split).")

    # scale so L2 norm is snr:
    # if vector has 25 entries of +/- a, norm = sqrt(25)*a => a = snr / 5
    a = snr / 5.0

    m1 =  a * np.ones(p_signal)
    m4 = -a * np.ones(p_signal)

    m2 =  a * np.r_[np.ones(13), -np.ones(12)]
    m3 =  a * np.r_[-np.ones(13),  np.ones(12)]

    return np.stack([m1, m2, m3, m4], axis=0)  # shape (4, 25)


def simulate_sparse_ar1_gmm(
    N=500,
    M=5000,
    K=4,
    cluster_sizes=(20, 80, 120, 280),
    p_signal=25,
    rho=0.5,
    snr=4.0,
    seed=0,
):
    """
    Simulate X ~ mixture of Gaussians with AR(1) covariance across features.
    First p_signal features are signal; remaining are noise (mean 0).
    """
    if sum(cluster_sizes) != N:
        raise ValueError(f"cluster_sizes must sum to N={N}. Got {sum(cluster_sizes)}.")
    if K != len(cluster_sizes):
        raise ValueError("K must match len(cluster_sizes).")
    if M < p_signal:
        raise ValueError("M must be >= p_signal.")

    rng = np.random.default_rng(seed)

    # means for signal block
    mu_signal = make_signal_means(K=K, p_signal=p_signal, snr=snr)  # (K, 25)

    X_list = []
    y_list = []

    start = 0
    for k, nk in enumerate(cluster_sizes):
        # sample AR(1) noise with covariance rho^{|i-j|}
        Xk = ar1_gaussian(nk, M, rho=rho, rng=rng)

        # add mean: signal dims only
        Xk[:, :p_signal] += mu_signal[k]

        X_list.append(Xk)
        y_list.append(np.full(nk, k, dtype=int))
        start += nk

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # optional shuffle
    perm = rng.permutation(N)
    return X[perm], y[perm]


def simulate_sparse_ar1_gmm(
    N=500,
    M=5000,
    K=4,
    cluster_sizes=(20, 80, 120, 280),
    p_signal=25,
    rho=0.5,
    snr=4.0,
    seed=0,
):
    """
    Simulate X ~ mixture of Gaussians with AR(1) covariance across features.
    First p_signal features are signal; remaining are noise (mean 0).
    """
    if sum(cluster_sizes) != N:
        raise ValueError(f"cluster_sizes must sum to N={N}. Got {sum(cluster_sizes)}.")
    if K != len(cluster_sizes):
        raise ValueError("K must match len(cluster_sizes).")
    if M < p_signal:
        raise ValueError("M must be >= p_signal.")

    rng = np.random.default_rng(seed)

    # means for signal block
    mu_signal = make_signal_means(K=K, p_signal=p_signal, snr=snr)  # (K, 25)

    X_list = []
    y_list = []

    start = 0
    for k, nk in enumerate(cluster_sizes):
        # sample AR(1) noise with covariance rho^{|i-j|}
        Xk = ar1_gaussian(nk, M, rho=rho, rng=rng)

        # add mean: signal dims only
        Xk[:, :p_signal] += mu_signal[k]

        X_list.append(Xk)
        y_list.append(np.full(nk, k, dtype=int))
        start += nk

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # optional shuffle
    perm = rng.permutation(N)
    return X[perm], y[perm]


def simulate_ar1_gmm_for_grid(N, p, K, cluster_sizes, rho, snr, p_signal=25, seed=0):
    p_signal = min(p_signal, p)
    X, y = simulate_sparse_ar1_gmm(
        N=N, M=p, K=K, cluster_sizes=cluster_sizes,
        p_signal=p_signal, rho=rho, snr=snr, seed=seed
    )
    true_idx = np.arange(p_signal)
    return X, y, true_idx, p_signal, p - p_signal