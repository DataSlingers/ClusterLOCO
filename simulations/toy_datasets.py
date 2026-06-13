"""
Toy datasets and particular cases used in paper.

- Grant simulations

"""
import numpy as np
from sklearn.datasets import *

################## GRANT: ANIRBAN 2D EXAMPLE GMM

K = 5
p = 2

np.random.seed(7)          # 2

Theta_x = np.linspace(-1.5, 1.5, K)
Theta_y = [1.0, -1.5, 0.0, 1.5, -0.5]

Theta = np.array([Theta_x, Theta_y]).T
Vars = np.random.rand(K, p)*0.52        # 0.65
weights = [4,2,2,2,1] # np.random.rand(K)
weights = weights / np.sum(weights)


def generate_data(n_samples=2000):
    labels = np.random.choice(K, size=n_samples, p=weights)
    data = []
    for i in range(n_samples):
        theta = Theta[labels[i]]
        var = Vars[labels[i]]
        sample = np.random.multivariate_normal(theta, np.diag(var))
        data.append(sample)
    return np.array(data), labels

###### USAGE Example: 
# X, y = generate_data()

# fig, ax = plt.subplots(figsize=(5, 5))
# for label in range(K):
#     mask = (y == label)
#     ax.scatter(X[mask, 0], X[mask, 1],
#                   color=dark_colors[label],
#                   s=15, alpha=0.5)
# ax.scatter(Theta[:, 0], Theta[:, 1], c='red', marker='x', s=20, label='Mixture Centers')
# ax.set_aspect('equal', 'datalim')
# ax.legend()
# # plt.savefig('loco_cluster_example.pdf', bbox_inches='tight')
# plt.show()


################## Grant example version with 10 D from 2 D

K = 5
p = 2

np.random.seed(42)

Theta_x = np.linspace(-1, 1, K)
Theta_y = np.random.randn(K) * 0.5

Theta = np.array([Theta_x, Theta_y]).T
Vars = np.random.rand(K, p)*0.1
weights = np.random.rand(K)
weights = weights / np.sum(weights)


# def generate_data(n_samples=1000, seed=123):
#     rng = np.random.RandomState(seed)
#     labels = np.random.choice(K, size=n_samples, p=weights)
#     data = []
#     for i in range(n_samples):
#         theta = Theta[labels[i]]
#         var = Vars[labels[i]]
#         sample = rng.multivariate_normal(theta, np.diag(var))
#         data.append(sample)
#     return np.array(data), labels



# from clim.cluster_generation import project_HD_per_cluster
def fixed_gaussian_projected(n_samples, seed, d, noise_d, Z):
    assert Z[0].shape[0] == d, "Z has wrong shape, not compatible with signal dimension d"
    X_sig = np.empty((n_samples, d), dtype=float)
    # Get YH's data in 2d 
    X_2d, y_2d = generate_data(n_samples=n_samples, seed=seed)

    # Project each cluster 
    for lb in np.unique(y_2d):
        mask = (y_2d == lb)
        if not np.any(mask):
            continue
        Z_k = Z[lb]
        X_sig[mask] = X_2d[mask] @ Z_k.T  # (n_k, r) @ (r, d) -> (n_k, d)

    rng = np.random.RandomState(seed)
    X_noise = rng.normal(size=(n_samples, noise_d))
    X_full = np.concatenate([X_sig, X_noise], axis=1)
    return X_full, y_2d

#### USAGE EXAMPLE WITH THE PARTICULAR Z GENERATED: 
# Z = np.load('../../results/Z_projection.npy', allow_pickle=True).item()

# X_2d, y_2d = generate_data()
# X, y, _ = project_HD_per_cluster(5,5, X_2d,y_2d,noise_type='gaussian',rng=None,per_cluster=True,preset_Z=Z

# fig = plt.figure(figsize=(5, 4.5))
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap='viridis', s=10)
# plt.scatter(Theta[:, 0], Theta[:, 1], c='red', marker='x', s=20, label='Mixture Centers')
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# plt.legend()
# plt.xlim((-2,2))
# plt.ylim((-2,2))
# plt.show()

############# Usual SKLEANR TOY DATASETS 

def sklearn_datasets(name="moons", n_samples=500, n_noise=4, SNR=0.05, seed=30, factor=0.8):
    rng = np.random.RandomState(seed)
    if name == "moons":
        X, y = datasets.make_moons(n_samples=n_samples, noise=SNR, random_state=seed)
    elif name == "circles":
        X, y = datasets.make_circles(n_samples=n_samples, noise=SNR, random_state=seed, factor=factor)
    elif name == "varied":
        X, y, centers = datasets.make_blobs(n_samples=n_samples, cluster_std=SNR * np.array([40, 10, 2]), random_state=seed, return_centers=True)
        noise = rng.normal(size=(X.shape[0], n_noise))
        X = np.concatenate([X, noise], axis=1)
        return X, y, centers
        
    else:
        raise ValueError("Unknown dataset name")

    noise = rng.normal(size=(X.shape[0], n_noise))
    X = np.concatenate([X, noise], axis=1)
    return X, y



######## INTERLACED HALFMOONS WITH NON LINEAR TRANSFORMATION FOR FIGURE 1

def make_interlaced_halfmoons_3feat(n_per_class=400, noise_xy=0.08, noise_f3=0.08, seed=0,
    r_outer=1.35, r_mid=1.00, r_inner=0.70, # arc geometry
    c_outer=(0.50, 0.35), c_mid=(-0.15, -0.05), c_inner=(0.55, -0.05), # centers in (x,y)
    ang_outer=(0.05*np.pi, 1.05*np.pi),   # top-ish arc
    ang_mid=(0.95*np.pi, 1.95*np.pi),     # bottom-ish arc
    ang_inner=(-0.15*np.pi, 1.15*np.pi),   # inner arc
    # feature-3 structure
    f3_mode="nonlinear",               # "class_offset" or "nonlinear"
    f3_offsets=(0.9, -0.7, 0.6),          # per-class shift for Feature 3
):
    """
    Returns
    -------
    X : (N,3) array
    y : (N,) int labels in {0,1,2}
    """

    rng = np.random.default_rng(seed)

    def sample_arc(n, r, center, ang_lo, ang_hi):
        t = rng.uniform(ang_lo, ang_hi, size=n)
        x = center[0] + r * np.cos(t)
        y = center[1] + r * np.sin(t)
        x += rng.normal(scale=noise_xy, size=n)
        y += rng.normal(scale=noise_xy, size=n)
        return x, y, t

    # Class 0: outer arc
    x0, y0, t0 = sample_arc(n_per_class, r_outer, c_outer, *ang_outer)
    # Class 1: mid arc (opposite side)
    x1, y1, t1 = sample_arc(n_per_class, r_mid,   c_mid,   *ang_mid)
    # Class 2: inner arc
    x2, y2, t2 = sample_arc(n_per_class, r_inner, c_inner, *ang_inner)

    X_xy = np.vstack([
        np.c_[x0, y0],
        np.c_[x1, y1],
        np.c_[x2, y2],
    ])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class, dtype=int)

    # Build Feature 3 to give a different projection
    if f3_mode == "class_offset":
        f3 = (0.55 * X_xy[:, 0] - 0.15 * X_xy[:, 1])  # shared linear trend
        f3 += np.array([f3_offsets[c] for c in y])    # class-dependent offset
        f3 += rng.normal(scale=noise_f3, size=len(y))
    elif f3_mode == "nonlinear":
        # nonlinear f3 that depends on radius/angle, plus small class shift
        r = np.sqrt((X_xy[:,0] - 0.1)**2 + (X_xy[:,1] + 0.05)**2)
        sin_term = np.sin(1.5 * r)
        log_term = np.log(np.abs(1 + X_xy[:,0]))
        sin_mult = np.ones(len(y))
        sin_mult[y == 0] = -1.0
        log_mult = np.full(len(y), 0.2)
        log_mult[y == 2] = 0.5
        f3 = 0.8 * sin_mult * sin_term + log_mult * log_term
        f3 += np.array([f3_offsets[c] for c in y])
        f3 += rng.normal(scale=noise_f3, size=len(y))
        
    else:
        raise ValueError("f3_mode must be 'class_offset' or 'nonlinear'")
    f3 = f3 - (np.cov(X_xy[:,0], f3, bias=True)[0,1] / np.var(X_xy[:,0])) * X_xy[:,0]
    X = np.c_[X_xy, f3]
    return X, y


###### Trajectory clustering 

def cluster_trajectory(n_samples, n_clusters, cluster_means = None, cluster_sigma = 0.1, path_noise_sigma=0.1):
    # Sample t_x using multimodal Gaussian
    M = np.random.choice(cluster_means, size=n_samples)
    Z = np.random.normal(0, cluster_sigma, size=n_samples)
    tx = norm.cdf(Z + M)  # ensures values are in [0,1]
    mean_dict = {mean: i for i, mean in enumerate(cluster_means)}
    labels = np.array([mean_dict[m] for m in M])
    
    # Define smooth path τ_x(t): A 2D spiral or S-curve
    def tau_x(t):
        # An S-curve in 2D
        return np.stack([t, np.sin(2 * np.pi * t)], axis=1)
    
    # Compute τ_x(t)
    X_mean = tau_x(tx)
    
    # Compute normal vectors numerically
    def compute_normals(t):
        eps = 1e-5
        tau_t = tau_x(t)
        tau_t_plus = tau_x(t + eps)
        tangent = (tau_t_plus - tau_t) / eps
        # Rotate tangent vector by 90 degrees to get normal
        normal = np.stack([-tangent[:,1], tangent[:,0]], axis=1)
        # Normalize
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        return normal
    
    # Generate noise in normal direction
    normals = compute_normals(tx)
    eps = np.random.normal(0, path_noise_sigma, size=(n_samples, 1))
    X = X_mean + eps * normals

    return X, tx, labels
    