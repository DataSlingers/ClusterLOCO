"""
Author: Claire He 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import qr, cholesky
from numpy.linalg import matrix_rank, solve
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from .mixture_models import *
from scipy.stats import chi2
from typing import Optional, Sequence
import scipy
from scipy.stats import chi2
from scipy.stats import norm, gamma as gamma_dist

def permute_feature(X, j, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Xp = X.copy()
    Xp[:, j] = rng.permutation(Xp[:, j])
    return Xp[:,j]
    
def project_HD(d, noise_d, samples_to_project, labels, noise_type='gaussian', random_state=42, Z_dict=False):
    """ Linearly project in HD using orthonormal projection """
    rng = np.random.RandomState(random_state)
    N, r = samples_to_project.shape
    # cluster assignment O-H encoded
    K = len(np.unique(labels))
    F = np.zeros((N, K))
    F[np.arange(N), labels] = 1.0
    # projection matrix to R^{d}
    Z_raw = rng.normal(size=(d, r))
    Q_mat, _ = qr(Z_raw)
    Z = Q_mat[:, :r]
    
    W = samples_to_project # (N, r)
    X = W @ Z.T # (N, d)
    
    # Concatenate pure noise
    if noise_type=='gaussian':
        X = np.concatenate([X, rng.normal(size=(N, noise_d))], axis=1)
    elif noise_type=='gamma':
        X = np.concatenate([X, rng.gamma(1, 1, size=(N, noise_d))], axis=1)
    if Z_dict:
        return X, labels, Z
    else:
        return X, labels


def project_HD_per_cluster(d, noise_d, samples_to_project, labels, noise_type='gaussian', rng=None, per_cluster=True, preset_Z=None):
    """
    Project latent samples (N x r) into R^d with cluster-specific orthonormal projections.
    
    Parameters
    ----------
    d : int
        Target signal dimension.
    noise_d : int
        Extra noise dimensions to append (total output dim = d + noise_d).
    samples_to_project : (N, r) array
        Latent samples to project.
    labels : (N,) array-like
        Cluster labels for each sample. Determines which Z_k to use.
    noise_type : {'gaussian','gamma',None}
        Type of noise appended in the last noise_d dims.
    rng : np.random.Generator or None
        Random generator; if None, a default is created.
    per_cluster : bool
        If True, use one Z_k per cluster. If False, use a single shared Z.
    preset_Z : dict or None
        Optional pre-specified projection matrices per label: {label: Z_k (d x r)}.
        If provided for a label, it's used instead of sampling a new one.

    Returns
    -------
    X_full : (N, d + noise_d) array
        Projected data with noise dims concatenated (if noise_d > 0).
    labels : (N,) array
        Echoed labels.
    Z_dict : dict
        Mapping {label: Z_k} used for projection (helpful for reproducibility).
    """
    if rng is None:
        rng = np.random.default_rng()

    X_lat = np.asarray(samples_to_project, dtype=float)
    labels = np.asarray(labels)
    N, r = X_lat.shape
    if r > d:
        raise ValueError(f"Latent dim r={r} must be <= target dim d={d} to form orthonormal Z (got r>d).")

    # unique labels in deterministic order of appearance
    unique_labels = []
    seen = set()
    for lb in labels:
        if lb not in seen:
            unique_labels.append(lb)
            seen.add(lb)

    # build per-cluster or shared projection(s)
    Z_dict = {}
    if per_cluster:
        for lb in unique_labels:
            if preset_Z is not None and lb in preset_Z:
                Z_k = np.asarray(preset_Z[lb], dtype=float)
                if Z_k.shape != (d, r):
                    raise ValueError(f"preset_Z[{lb}] must have shape (d, r)=({d},{r}), got {Z_k.shape}")
            else:
                Z_raw = rng.standard_normal((d, r))
                Q, _ = np.linalg.qr(Z_raw, mode='reduced')  # Q: (d, r) with orthonormal columns
                Z_k = Q                                    # already d x r
            Z_dict[lb] = Z_k
    else:
        # single shared Z
        if preset_Z is not None:
            # allow preset under key None or first label
            key = None if None in preset_Z else unique_labels[0]
            Z_shared = np.asarray(preset_Z[key], dtype=float)
            if Z_shared.shape != (d, r):
                raise ValueError(f"preset_Z[{key}] must have shape (d, r)=({d},{r}), got {Z_shared.shape}")
        else:
            Z_raw = rng.standard_normal((d, r))
            Q, _ = np.linalg.qr(Z_raw, mode='reduced')
            Z_shared = Q
        for lb in unique_labels:
            Z_dict[lb] = Z_shared

    # project cluster by cluster, preserving original order
    X_sig = np.empty((N, d), dtype=float)
    for lb in unique_labels:
        mask = (labels == lb)
        if not np.any(mask):
            continue
        Z_k = Z_dict[lb]
        X_sig[mask] = X_lat[mask] @ Z_k.T  # (n_k, r) @ (r, d) -> (n_k, d)

    # append noise dims
    if noise_d > 0:
        if noise_type == 'gaussian':
            X_noise = rng.normal(size=(N, noise_d))
        elif noise_type == 'gamma':
            X_noise = rng.gamma(shape=1.0, scale=1.0, size=(N, noise_d))
        elif noise_type is None:
            X_noise = np.zeros((N, noise_d))
        else:
            raise ValueError("noise_type must be 'gaussian', 'gamma', or None")
        X_full = np.concatenate([X_sig, X_noise], axis=1)
    else:
        X_full = X_sig

    return X_full, labels, Z_dict

def _circle_overlap_area(d, r1, r2):
    """
    Exact area of overlap between two circles of radii r1, r2 with center distance d.
    Returns 0 if disjoint, min(area) if one contains the other, else lens area.
    """
    # Disjoint
    if d >= r1 + r2:
        return 0.0
    # One inside the other
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2)**2
    # Partial overlap (lens)
    r1_2, r2_2 = r1*r1, r2*r2
    alpha = np.arccos((d*d + r1_2 - r2_2) / (2*d*r1)) * 2
    beta  = np.arccos((d*d + r2_2 - r1_2) / (2*d*r2)) * 2
    area1 = 0.5 * r1_2 * (alpha - np.sin(alpha))
    area2 = 0.5 * r2_2 * (beta  - np.sin(beta))
    return area1 + area2

def random_circles(
    n_samples=1000,
    n_clusters=4,
    radii=None,                 # list/array of length K or None (random)
    half_circle_prob=0.5,
    noise=0.05,
    max_overlap_pct=5.0,        # p% cap on pairwise overlap (treat half-moon as full circle)
    bbox=(-6, 6, -6, 6),        # (xmin, xmax, ymin, ymax) placement region
    centers=None,               # optional preset centers (overrides placement if given)
    max_place_tries=10_000,     # global tries to place all centers
    per_center_tries=500,       # tries per center before bailing
    rng=None,
    diag=False                  # return out diagnostics
):
    """
    Generate K clusters in 2D: each is either a full circle or a half-circle with random orientation.
    Enforces that for any pair (i,j), circle overlap area ≤ (max_overlap_pct/100) * min(area_i, area_j).

    Overlap is computed using the FULL circles even for half-circles (your rule).
    """
    if rng is None:
        rng = np.random.default_rng()

    K = n_clusters
    # radii
    if radii is None:
        radii = rng.uniform(0.5, 1.5, size=K)
    else:
        radii = np.asarray(radii, dtype=float)
        assert radii.shape == (K,), "radii must have length n_clusters"

    # decide half vs full and orientation
    is_half = rng.random(K) < half_circle_prob
    orientations = rng.uniform(0, 2*np.pi, size=K)

    # placement region
    xmin, xmax, ymin, ymax = bbox
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        assert centers.shape == (K, 2), "centers must be shape (K,2)"
    else:
        centers = np.full((K, 2), np.nan, dtype=float)

        placed = 0
        tries = 0
        max_overlap_frac = max_overlap_pct / 100.0

        while placed < K and tries < max_place_tries:
            tries += 1
            # try to place next center
            ok = False
            for _ in range(per_center_tries):
                # sample a candidate center; keep the full circle inside bbox with a small margin
                r = radii[placed]
                cx = rng.uniform(xmin + r, xmax - r)
                cy = rng.uniform(ymin + r, ymax - r)
                # check overlap with previously placed
                viol = False
                for j in range(placed):
                    d = np.hypot(cx - centers[j,0], cy - centers[j,1])
                    A = _circle_overlap_area(d, r, radii[j])
                    A_cap = max_overlap_frac * np.pi * min(r, radii[j])**2
                    if A > A_cap:
                        viol = True
                        break
                if not viol:
                    centers[placed] = (cx, cy)
                    ok = True
                    break
            if ok:
                placed += 1
            else:
                # if we failed to place this center after many tries, randomize earlier ones a bit
                # (simple restart strategy)
                placed = max(0, placed - 1)
                if placed > 0:
                    centers[placed:] = np.nan
        if placed < K:
            raise RuntimeError(
                f"Could not place {K} circles within overlap cap {max_overlap_pct}% "
                f"in bbox {bbox}. Try enlarging bbox, shrinking radii, or raising the cap."
            )

    # sample counts
    counts = np.full(K, n_samples // K, dtype=int)
    counts[: n_samples % K] += 1

    # sample points on each (half/full) circle
    X_parts = []
    y_parts = []
    for k in range(K):
        n_k = counts[k]
        r = radii[k]
        cx, cy = centers[k]
        theta = rng.uniform(0, np.pi, size=n_k) if is_half[k] else rng.uniform(0, 2*np.pi, size=n_k)
        theta = theta + orientations[k]  # rotate arc

        x = cx + r * np.cos(theta) + rng.normal(scale=noise, size=n_k)
        y = cy + r * np.sin(theta) + rng.normal(scale=noise, size=n_k)

        X_parts.append(np.column_stack([x, y]))
        y_parts.append(np.full(n_k, k, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    meta = {
        "centers": centers,
        "radii": radii,
        "is_half": is_half,
        "orientations": orientations,
        "bbox": bbox,
        "max_overlap_pct": max_overlap_pct,
    }
    if diag:
        return X, y, meta
    return X, y

class GenerateCovariances:
    def __init__(self,
                 dim,
                 covMethod='eigen',
                 eigenval=None,
                 alpha_d=1,
                 eta=1.0,
                 rangeVar=(1.0, 10.0),
                 eigenmin=1.0,
                 eigenratio=10.0,
                 random_state=None):
        """
        Choice of covariance method in covMethod: 
          - 'eigen': random eigenvalues + random orthogonal eigenvectors
          - 'onion': Joe’s onion method for correlation + uniform variances
          - 'cvine' : Joe’s C‐vine method for correlation + uniform variances

        Parameters
        ----------
        dim : int
            Dimension of the covariance matrix.
        covMethod : str
            One of {'eigen', 'onion', 'cvine'}.
        eigenval : array‐like or None
            If provided, used directly as the eigenvalues in 'eigen' mode.
        eta : float
            Shape parameter for onion/cc-vine (must be > 0).
        rangeVar : tuple(float, float)
            (min_variance, max_variance) for sampling variances.
        eigenmin : float
            Minimum eigenvalue in 'eigen' mode.
        eigenratio : float
            Ratio max_eigenvalue/min_eigenvalue in 'eigen' mode.
        random_state : int or None
            Seed for reproducibility.
        """
        # store arguments
        self.dim         = int(dim)
        self.method      = covMethod.lower()
        self.eigenval    = eigenval
        self.eta         = float(eta)
        self.lb, self.ub = float(rangeVar[0]), float(rangeVar[1])
        self.eigenmin    = float(eigenmin)
        self.eigenratio  = float(eigenratio)
        self.random_state= random_state

        # RNG
        self.rng = np.random.default_rng(self.random_state)

        # basic sanity checks
        assert self.dim >= 1, "dim must be ≥ 1"
        assert self.lb < self.ub, "lower bound must be < upper bound"
        assert self.lb > 0, "variance lower bound must be > 0"
        assert self.eigenratio >= 1, "eigenratio must be ≥ 1"
        assert self.eigenmin > 0, "eigenmin must be > 0"
        assert self.eta > 0, "eta must be > 0"

        # placeholders
        self.Cov       = None
        self.eigvalues = None

    def covGen(self):
        """ Generate self.Cov and self.eigvalues according to self.method. """
        if self.method == 'eigen':
            self.Cov = self._eigencovariance()
            self.eigvalues, self.eigvectors = np.linalg.eigh(self.Cov)

        elif self.method == 'onion':
            R = self._onioncovariance()
            sigma2 = self.rng.uniform(self.lb, self.ub, size=self.dim)
            D = np.diag(np.sqrt(sigma2)) if self.dim > 1 else np.sqrt(sigma2)
            self.Cov = D @ R @ D
            self.eigvalues, self.eigvectors = np.linalg.eigh(self.Cov)

        elif self.method in ('c-vine', 'cvine', 'vine'):
            R = self._cvinecovariance()
            sigma2 = self.rng.uniform(self.lb, self.ub, size=self.dim)
            D = np.diag(np.sqrt(sigma2)) if self.dim > 1 else np.sqrt(sigma2)
            self.Cov = D @ R @ D
            self.eigvalues, self.eigvectors = np.linalg.eigh(self.Cov)
          

        else:
            raise ValueError(f"Unknown covMethod: {self.method!r}")

        return self.Cov, self.eigvalues

    def _orthogonal_matrix(self):
        """ Generate a random orthogonal matrix Q of shape (dim, dim). """
        # draw a random Gaussian matrix and QR‐decompose it
        A = self.rng.standard_normal((self.dim, self.dim))
        Q, _ = scipy.linalg.qr(A)
        # enforce det(Q)=+1
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q

    def _eigencovariance(self):
        """ Random‐eigenvalue covariance. """
        # pick eigenvalues
        if self.eigenval is None:
            low = self.eigenmin
            high= self.eigenmin * self.eigenratio
            vals= self.rng.uniform(low, high, size=self.dim)
        else:
            vals = np.asarray(self.eigenval, dtype=float)
            if vals.shape != (self.dim,):
                raise ValueError("eigenval must have length dim")

        # form diagonal & rotate
        U = np.diag(vals)
        if self.dim > 1:
            Q = self._orthogonal_matrix()
            return Q @ U @ Q.T
        else:
            return U

    def _onioncovariance(self):
        """ Joe’s onion method for random correlation matrix. """
        d, eta, rng = self.dim, self.eta, self.rng

        # trivial dims
        if d == 1:
            return np.array([[1.0]])
        if d == 2:
            rho = 2 * rng.beta(eta, eta) - 1
            return np.array([[1.0, rho],
                             [rho, 1.0]])

        R    = np.eye(d)
        beta = eta + (d - 2) / 2.0

        # first off‐diagonal
        r12      = 2 * rng.beta(beta, beta) - 1
        R[0,1:]  = r12
        R[1:,0]  = r12

        # grow from 2…d-1
        for m in range(2, d):
            beta = beta - 0.5
            y    = rng.beta(m/2.0, beta)
            z    = rng.standard_normal(m)
            z   /= np.linalg.norm(z)
            w     = np.sqrt(y) * z

            L = np.linalg.cholesky(R[:m, :m])
            U = L.T
            qq= w @ U

            R[:m, m] = qq
            R[m, :m] = qq
            R[m, m]  = 1.0

        return R

    def _cvinecovariance(self):
        """ Joe’s C‐vine method for random correlation matrix. """
        d, eta, rng = self.dim, self.eta, self.rng

        # trivial dims
        if d == 1:
            return np.array([[1.0]])
        if d == 2:
            rho = 2 * rng.beta(eta, eta) - 1
            return np.array([[1.0, rho],
                             [rho, 1.0]])

        R = np.eye(d)
        P = np.zeros((d, d))

        # first row
        beta0 = eta + (d-2)/2.0
        for j in range(1, d):
            rho = 2 * rng.beta(beta0, beta0) - 1
            R[0, j] = R[j, 0] = P[0, j] = rho

        # fill in the rest
        for m in range(1, d-1):
            alpha = eta + (d - 2 - m) / 2.0
            for i in range(m+1, d):
                rho = 2 * rng.beta(alpha, alpha) - 1
                tem = rho
                # back‐solve through lower‐order partials
                for k in range(m-1, -1, -1):
                    a = P[k, m]
                    b = P[k, i]
                    tem = a*b + tem * np.sqrt((1 - a*a) * (1 - b*b))
                R[m, i] = R[i, m] = tem
                P[m, i] = rho

        return R



def random_polynomial_features(X, embed_dim, degree, rng = None, include_linear=True, include_bias = False, degree_probs = None, allow_powers = True, standardize = True, eps = 1e-12):
    """
    Randomly sample `embed_dim` polynomial (monomial) features up to `degree` without enumerating all.
    Each sampled feature is a monomial:  prod_{t=1..k} X[:, idx_t]
      - if allow_powers=True, idx_t sampled WITH replacement -> powers allowed (x_i^p)
      - if allow_powers=False, idx_t sampled WITHOUT replacement -> pure interactions

    Parameters
    ----------
    X : (n, d)
    embed_dim : number of sampled monomials (not counting optional linear/bias)
    degree : maximum monomial degree (>=1)
    rng : numpy Generator
    include_linear : if True, include original X as features
    include_bias : if True, prepend a column of ones
    degree_probs : probabilities over degrees 1..degree (len=degree). If None, uniform.
                  Example for favoring low degrees: degree_probs=[0.6, 0.3, 0.1]
    allow_powers : if True, allow repeated indices -> powers
    standardize : if True, z-score X before building monomials (helps numeric stability)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if embed_dim < 0:
        raise ValueError("embed_dim must be >= 0")
    if not allow_powers and degree > d:
        raise ValueError("If allow_powers=False, need degree <= d (can't choose unique indices).")

    if rng is None:
        rng = np.random.default_rng()

    Xw = X
    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        Xw = (X - mu) / sd

    # Choose degrees for each sampled monomial
    if degree_probs is None:
        degree_probs = np.ones(degree) / degree
    else:
        degree_probs = np.asarray(degree_probs, dtype=float)
        if degree_probs.shape != (degree,):
            raise ValueError("degree_probs must have length == degree")
        s = degree_probs.sum()
        if s <= 0:
            raise ValueError("degree_probs must sum to > 0")
        degree_probs = degree_probs / s

    degs = rng.choice(np.arange(1, degree + 1), size=embed_dim, replace=True, p=degree_probs)

    # Compute sampled monomials
    Phi = np.ones((n, embed_dim), dtype=float)
    for j, k in enumerate(degs):
        if allow_powers:
            idx = rng.integers(0, d, size=k)  # with replacement
        else:
            idx = rng.choice(d, size=k, replace=False)  # without replacement
        # product along chosen indices
        Phi[:, j] = np.prod(Xw[:, idx], axis=1)

    parts = []
    if include_bias:
        parts.append(np.ones((n, 1), dtype=float))
    if include_linear:
        parts.append(Xw)
    parts.append(Phi)

    return np.concatenate(parts, axis=1)



class BaseSimulator:
    def __init__(self, K, n_samples_per_cluster, alpha=1.0, d_0=4, Cov_k=None, method='gaussian', center_method='random', random_state=None, mode='original', gaps=None, r=None, balanced=True, oversample=2.0, per_cluster_seed=True, shape_probs=None):
        """
        Simulator class for cluster data. 

        """
        self.K = int(K)
        self.alpha = float(alpha)
        self.n = int(K)*n_samples_per_cluster
        self.d_0 = int(d_0)
        self.method = method.lower()
        self.mode = mode.lower()
        self.center_method = center_method
        self.gaps = gaps
        self.r = r
        if shape_probs is None:
            self.shape_probs = {"donut": 0.5, "moon": 0.5}
        else:
            self.shape_probs = shape_probs

        # samples per cluster: int or array-like
        if np.isscalar(n_samples_per_cluster):
            self.nk = np.full(self.K, int(n_samples_per_cluster), dtype=int)
        else:
            nk = np.asarray(n_samples_per_cluster, dtype=int)
            if nk.shape != (self.K,):
                raise ValueError("n_samples_per_cluster must be int or shape (K,)")
            self.nk = nk

        self.balanced = bool(balanced)
        self.oversample = float(oversample)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.per_cluster_seed = bool(per_cluster_seed)

        # covariance handling + precompute cholesky
        if Cov_k is None:
            self.Cov_k = np.repeat(np.eye(self.d_0)[None, :, :], self.K, axis=0)
        else:
            cov_arr = np.asarray(Cov_k, dtype=float)
            if cov_arr.shape != (self.K, self.d_0, self.d_0):
                raise ValueError("Cov_k must have shape (K, d_0, d_0)")
            self.Cov_k = cov_arr

        # Precompute transforms once
        self.L_k = np.empty_like(self.Cov_k)
        for k in range(self.K):
            self.L_k[k] = cholesky(self.Cov_k[k])

        self.cluster_centers = None
        self.X = None
        self.labels = None
        
    def _find_vertices(self, p):
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
        
    # ---------- centers ----------
    def cluster_gen(self, center_method=None):
        if center_method is None:
            C0 = self._generate_simplex_centroids(self.K, self.d_0)  # unit-scale
        elif center_method == 'random':
            C0 = self.alpha * self.rng.standard_normal((self.K, self.d_0))

        else:
            raise ValueError(f"Unknown center_method '{center_method}'")

        centers = C0
        self.cluster_centers = centers  # not scaled here

    def _generate_simplex_centroids(self, K, d_0):
        """
        Generate K centroids in r dimensions using simplex + shifting logic.
        Used by both cluster_gen() and low_rank_gmm().
        """
        base = self._find_vertices(d_0) # find all vertices in simplex, will be (d_0 + 1, d_0)

        cluster_centers = np.zeros((K, d_0), dtype=float)
        # if K <= d_0+1
        n0 = min(K, d_0 + 1)
        cluster_centers[:n0, :] = base[:n0, :]
        if K > d_0 + 1:
            e1 = np.zeros(d_0, dtype=float)
            e1[0] = 1.0
        
            # How many *full* blocks of d_0 points beyond the first d_0+1?
            full_blocks = (K - (d_0 + 1)) // d_0
            remainder   = (K - (d_0 + 1)) % d_0
        
            idx = d_0 + 1
            # For each full block j = 1 … full_blocks, shift by 2 * j * e1
            for j in range(1, full_blocks + 1):
                for i in range(1, d_0 + 1):
                    cluster_centers[idx, :] = base[i, :] + 2 * j * e1
                    idx += 1
                
            # Finally the leftover remainder (shift by 2*(full_blocks+1))
            if remainder > 0:
                shift = 2 * (full_blocks + 1)
                for i in range(1, remainder + 1):
                    cluster_centers[idx, :] = base[i, :] + shift * e1
                    idx += 1
    
        return self.alpha * cluster_centers

    def _sample_gamma_cluster(self, k, n, rng):
        scale = self.scale[k]
        shape = self.shape
        return rng.gamma(shape, scale, size=(n, self.d_0))
        

    # ---------- fast gaussian sampling ----------
    def _sample_gaussian_cluster(self, k, n, rng):
        Z = rng.standard_normal((n, self.d_0))
        return (self.alpha * self.cluster_centers[k]) + Z @ self.L_k[k].T

    # ---------- shape masks ----------
    def _intrinsic_chol(self, k):
        # Cholesky of intrinsic covariance (d0 x d0)
        idx = slice(0, self.d_0)
        C0 = self.Cov_k[k][idx, idx]
        return cholesky(C0)
        
    def _mask_donuts_intr(self, Xk, center, thickness, k, q=0.95):
        """
        thickness: fraction of outer radius to remove from inside.
          thickness=0.2 -> keep ring [0.8*r_out, r_out]
        """
        idx = slice(0, self.d_0)
        L0 = self._intrinsic_chol(k)
        Z = solve(L0, (Xk[:, idx] - center[idx]).T).T  # whitened coords
        R = np.linalg.norm(Z, axis=1)
    
        r_out = np.sqrt(chi2.ppf(q, df=self.d_0))
        r_in  = max(1e-12, (1.0 - float(thickness)) * r_out)
        return (R >= r_in) & (R <= r_out)
    
    def _mask_moons_intr(self, Xk, center, bite_frac, shift_frac, k, rng, q=0.95, u=None):
        """
        Creates a crescent ("moon") by cutting a bite out of the outer disk.
    
        If u is provided, it is used as the bite direction in whitened space.
        If u is None, a random direction is sampled (original behavior).
        """
        idx = slice(0, self.d_0)
        L0 = self._intrinsic_chol(k)
    
        Z = solve(L0, (Xk[:, idx] - center[idx]).T).T
        R0 = np.linalg.norm(Z, axis=1)
    
        r_out = np.sqrt(chi2.ppf(q, df=self.d_0))
        r_bite = float(bite_frac) * r_out
    
        # Direction in whitened space
        if u is None:
            u = rng.standard_normal(self.d_0)
    
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != self.d_0:
            raise ValueError(f"u must have length d_0={self.d_0}, got {u.size}")
    
        u /= (np.linalg.norm(u) or 1.0)
    
        shift = float(shift_frac) * r_out * u
        Rm = np.linalg.norm(Z - shift, axis=1)
    
        # Keep points inside outer disk but outside bite disk
        keep = (R0 <= r_out) & (Rm >= r_bite)
        return keep
        
    def _moon_params_from_gap(self, g):
        # g in [0,1] => bigger g => stronger crescent
        g = float(g)
        bite_frac  = 0.98 - 0.20 * g   # from 0.98 down to 0.78
        shift_frac = 0.40 + 0.90 * g   # from 0.40 up to 1.30
        return bite_frac, shift_frac

    def _rescale_center_separation(self, target_sep, axis=None):
        """
        Rescale the separation between two centers to target_sep while preserving direction.
        Only for K=2.
        """
        if self.K != 2:
            return
    
        c0, c1 = self.cluster_centers[0].copy(), self.cluster_centers[1].copy()
        v = c1 - c0
        dist = np.linalg.norm(v)
        if dist < 1e-12:
            # if coincident, pick an axis
            v = np.zeros_like(c0)
            v[0] = 1.0
            dist = 1.0
    
        v = v / dist
        mid = 0.5 * (c0 + c1)
    
        self.cluster_centers[0] = mid - 0.5 * target_sep * v
        self.cluster_centers[1] = mid + 0.5 * target_sep * v
    
    # ---------- oversample-until-enough ----------
    def _sample_shaped_cluster_exact(self, k, n_target, shape, rng, q=0.95):
        center = self.alpha * self.cluster_centers[k]
        g = float(self.gaps[k]) if self.gaps is not None else 0.5
    
        kept = []
        n_need = n_target
        batch = max(int(np.ceil(self.oversample * n_target)), n_target)
    
        # fixed moon directions for interlocking (works best for K=2)
        u = None
        if shape == "moon" and hasattr(self, "_moon_u") and k in self._moon_u:
            u = self._moon_u[k]
            
        # if shape == "moon":
            # u = np.zeros(self.d_0)
            # u[0] = 1.0
            # if self.K == 2:
            #     if k == 1:
            #         u = -u  # flip direction for the second moon
        
        while n_need > 0:
            Xb = self._sample_gaussian_cluster(k, batch, rng)
    
            if shape == "donut":
                keep = self._mask_donuts_intr(Xb, center=center, thickness=g, k=k, q=q)
            elif shape == "moon":
                bite_frac, shift_frac = self._moon_params_from_gap(g)
                keep = self._mask_moons_intr(
                    Xb, center=center, bite_frac=bite_frac, shift_frac=shift_frac,
                    k=k, rng=rng, q=q, u=u
                )
            else:
                keep = np.ones(Xb.shape[0], dtype=bool)
    
            Xk = Xb[keep]
            if Xk.shape[0] > 0:
                take = min(n_need, Xk.shape[0])
                kept.append(Xk[:take])
                n_need -= take
    
            if n_need > 0:
                batch = int(batch * 1.5) + 10
    
        return np.vstack(kept)
    
    def low_rank_gmm(self):
        """
        Low-rank Gaussian mixture with K clusters in first d_0 dims,
        then (d - d_0) pure noise dims appended.
    
        Conventions:
          - _generate_simplex_centroids returns UN-SCALED centers
          - alpha is applied ONCE here to the centroid matrix
        """
        if self.r is None:
            r = max(1, self.d_0 // 2)
        else:
            r = int(self.r)
    
        K = self.K
        d0 = self.d_0
        n_per = int(self.nk[0]) if hasattr(self, "nk") else int(self.n_samples)
        total = K * n_per
    
        labels = np.repeat(np.arange(K), n_per)
    
        # one-hot membership matrix F
        F = np.zeros((total, K), dtype=float)
        F[np.arange(total), labels] = 1.0
    
        # orthonormal Z in R^{d0 x r}
        Z_raw = self.rng.standard_normal((d0, r))
        Q_mat, _ = qr(Z_raw)
        Z = Q_mat[:, :r]
    
        # W in R^{K x r}
        if self.mode == "original":
            W = np.zeros((K, r), dtype=float)
            while matrix_rank(W) < r:
                W = self.rng.choice([-1.0, 0.0, +1.0], size=(K, r))
        else:
            # simplex centroids in r dims, UN-SCALED
            W = self._generate_simplex_centroids(K=K, d_0=r)
    
        # centroids Theta in R^{K x d0}
        Theta = (self.alpha * W) @ Z.T  # apply alpha once here
    
        # mean matrix B: (total x d0)
        B = F @ Theta
    
        # simulate intrinsic observations
        X0 = B + self.rng.standard_normal((total, d0))

    
        self.X = X0
        self.labels = labels
        self.cluster_centers = Theta

    def generate_data(self, **kwargs):
        if self.method in ("gaussian", "moon-donut", 'gamma'):
            self.cluster_gen(self.center_method)
            Xs = []
            ys = []
            for k in range(self.K):
                rngk = (np.random.default_rng(self.random_state + k)
                        if (self.random_state is not None and self.per_cluster_seed)
                        else self.rng)

                n = int(self.nk[k])
                if self.method == "gaussian":
                    Xk = self._sample_gaussian_cluster(k, n, rngk)
                    
                elif self.method == 'moon-donut':
                    shape = rngk.choice(["donut", "moon"], p=[self.shape_probs["donut"], self.shape_probs["moon"]])
                    Xk = self._sample_shaped_cluster_exact(k, n, shape, rngk)
                    target_sep = kwargs.get('target_sep', 0.35)
                    self._rescale_center_separation(target_sep)
                    
                elif self.method == "gamma":
                    self.shape = np.full(self.d_0, self.alpha)
                    self.scale = make_gamma_scales_sparse_rs(rng = rngk, K=self.K, d=self.d_0)

                    Xk = self._sample_gamma_cluster(k, n, rngk)
                
                Xs.append(Xk)
                ys.append(np.full(Xk.shape[0], k, dtype=int))

            self.X = np.vstack(Xs)
            self.labels = np.concatenate(ys)
            return self.X, self.labels

        elif self.method == "low-rank-gaussian":
            self.low_rank_gmm()
            return self.X, self.labels

        elif self.method == 'swiss-roll':
            phi, psi, labels = self._build_phi_psi_(nk = self.nk, d_0 = self.d_0, K = self.K, rng=self.rng)
            eta =  kwargs.get('eta', 1.0)
            X = self._embed_on_swiss_roll_(phi, psi, eta=eta, rng=self.rng)
            if self.d_0%2 ==0:
                X = np.delete(X, -2, axis=1)
            self.X = X
            self.labels = labels
            return self.X, self.labels

        else:
            raise ValueError(f"Unknown method '{self.method}'")

    def add_noise(self, noise_d, noise_type='gaussian', **kwargs):
        """
        Add noise dimensions can be using following distributions:
        {"gaussian", "student-t", "uniform", "triangular", "laplace"}.

        kwargs :
            Distribution-specific parameters:
                student-t : df (default=5)
                uniform   : low (default=-1), high (default=1)
                triangular: low (default=-1), high (default=1), mode (default=0)
                laplace   : scale (default=1.0)
        """
        n = self.n
        if noise_type == "gaussian":
            X_noise = self.rng.normal(size=(n, noise_d))
    
        elif noise_type == "student-t":
            df = kwargs.get("df", 5)
            X_noise = self.rng.standard_t(df=df, size=(n, noise_d))
    
        elif noise_type == "uniform":
            low = kwargs.get("low", -1.0)
            high = kwargs.get("high", 1.0)
            X_noise = self.rng.uniform(low=low, high=high, size=(n, noise_d))
    
        elif noise_type == "triangular":
            low = kwargs.get("low", -1.0)
            high = kwargs.get("high", 1.0)
            mode = kwargs.get("mode", 0.0)
            X_noise = self.rng.triangular(left=low, mode=mode, right=high, size=(n, noise_d))
    
        elif noise_type == "laplace":
            scale = kwargs.get("scale", 1.0)
            X_noise = self.rng.laplace(loc=0.0, scale=scale, size=(n, noise_d))
    
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")
    
        self.X = np.concatenate([self.X, X_noise], axis=1)
        return self.X

    def _build_phi_psi_(self, nk, d_0, K, rng):
        d = d_0//2
        prop = rng.choice([0, 1], size=d-1)
        centers = np.zeros((K, d))
        centers[:, 0] = np.sort(rng.uniform(low=1.5*np.pi, high=4.5*np.pi, size=K))
        rho = 1/self.alpha
        cov = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
        for i in range(1, d):
            if prop[i-1] == 0:
                centers[:, i] = np.linspace(-K/d, K/d, K)
            else:
                centers[:, i] = np.sort(rng.uniform(low=1.5*np.pi, high=4.5*np.pi, size=K))
        phi, psi, labels = [], [], []
        for k, center in enumerate(centers):
            samples = rng.multivariate_normal(mean=center, cov=cov, size=nk[k])
            height = rng.uniform(low=-np.max(center), high=np.max(center), size=nk[k])
            phi.append(samples)
            labels.append(np.full(nk[k], k, dtype=int))
            psi.append(height)
        phi = np.vstack(phi)
        psi = np.concatenate(psi)
        labels = np.concatenate(labels)
        return phi, psi, labels
                    
    
    def _stable_unique(self, labels):
        seen = set()
        out = []
        for lb in labels:
            if lb not in seen:
                out.append(lb)
                seen.add(lb)
        return out
        
    def _embed_on_swiss_roll_(self, phi, psi, eta, rng):
        x = phi * np.cos(phi)  + eta*rng.uniform(size=phi.shape)
        y = phi * np.sin(phi)  + eta*rng.uniform(size=phi.shape)
        X = np.concatenate([x, y, psi.reshape(-1, 1)], axis=1)
        return X
        
    def project_higher_dim(
        self,
        embed_dim,
        method="orthogonal",
        degree=3,
        gamma=0.5,
        weight_scale=1.0,
        activation="relu",
        label_aware=True,
        **kwargs,
    ):
        """
        If label_aware=True, applies a cluster-specific random feature map per label.
        Requires self.labels to be set (call generate_data() first).
        """
        X = np.asarray(self.X, dtype=float)
        n, d = X.shape
        embed_dim = int(embed_dim)
    
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
    
        if label_aware and (getattr(self, "labels", None) is None):
            raise ValueError("label_aware=True requires self.labels (call generate_data first).")
    
        method = method.lower()
    
        # -------- helpers to apply per cluster --------
        def apply_per_cluster(make_params_and_map, out_dim):
            y = np.asarray(self.labels)
            uniq = self._stable_unique(y)
    
            Z = np.empty((n, out_dim), dtype=float)
            params = {}
    
            for lb in uniq:
                mask = (y == lb)
                Xk = X[mask]
                pk, Zk = make_params_and_map(lb, Xk)
                params[lb] = pk
                Z[mask] = Zk
            return Z, params
    
        # -------- non label-aware path: one global map --------
        def global_map(Z):
            self.X = Z
            return self.X
    
        # ================= ORTHOGONAL =================
        if method == "orthogonal":
            if embed_dim <= d:
                raise ValueError("embed_dim must be > current dimension for orthogonal lift")
    
            if label_aware:
                # cluster-specific orthogonal lift: Q_k in R^{embed_dim x d}
                def make(lb, Xk):
                    A = self.rng.standard_normal((embed_dim, embed_dim))
                    Q, _ = np.linalg.qr(A)
                    Q_embed = Q[:, :d]            # (embed_dim, d) orthonormal rows
                    Zk = Xk @ Q_embed.T           # (n_k, embed_dim)
                    return {"Q_embed": Q_embed}, Zk
    
                Z, params = apply_per_cluster(make, embed_dim)
                self._embed_params = {"method": "orthogonal", "per_label": params}
    
                return global_map(Z)
    
            # non label-aware
            A = self.rng.standard_normal((embed_dim, embed_dim))
            Q, _ = np.linalg.qr(A)
            Q_embed = Q[:, :d]
            return global_map(X @ Q_embed.T)
    
        # ================= RFF =================
        if method == "rff":
            # RFF does NOT require embed_dim > d; keep your check if you want, but not needed
            def rff_map(Xk, W, b):
                return np.sqrt(2.0 / embed_dim) * np.cos(Xk @ W + b)
    
            if label_aware:
                def make(lb, Xk):
                    W = self.rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d, embed_dim))
                    b = self.rng.uniform(0.0, 2.0 * np.pi, size=(embed_dim,))
                    Zk = rff_map(Xk, W, b)
                    return {"W": W, "b": b}, Zk
    
                Z, params = apply_per_cluster(make, embed_dim)
                self._embed_params = {"method": "rff", "per_label": params}
                return global_map(Z)
    
            W = self.rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d, embed_dim))
            b = self.rng.uniform(0.0, 2.0 * np.pi, size=(embed_dim,))
            return global_map(rff_map(X, W, b))
    
        # ================= NEURAL =================
        if method == "neural":
            def act(H):
                if activation == "tanh":
                    return np.tanh(H)
                if activation == "relu":
                    return np.maximum(H, 0.0)
                if activation == "sigmoid":
                    return 1.0 / (1.0 + np.exp(-H))
                raise ValueError("activation must be one of: tanh, relu, sigmoid")
    
            if label_aware:
                def make(lb, Xk):
                    W = self.rng.normal(0.0, weight_scale / np.sqrt(d), size=(d, embed_dim))
                    b = self.rng.normal(0.0, 1.0, size=(embed_dim,))
                    Zk = act(Xk @ W + b)
                    return {"W": W, "b": b}, Zk
    
                Z, params = apply_per_cluster(make, embed_dim)
                self._embed_params = {"method": "neural", "per_label": params}
                return global_map(Z)
    
            W = self.rng.normal(0.0, weight_scale / np.sqrt(d), size=(d, embed_dim))
            b = self.rng.normal(0.0, 1.0, size=(embed_dim,))
            return global_map(act(X @ W + b))
    
        # ================= POLYNOMIAL =================
        if method == "polynomial":
            # assumes your random_polynomial_features samples embed_dim monomials
            # Make it label-aware by sampling different monomials / coefficients per cluster.
            def poly_map(Xk, rngk):
                Zk = random_polynomial_features(
                    Xk,
                    embed_dim=embed_dim,
                    degree=degree,
                    rng=rngk,
                    include_linear=False,
                    **kwargs,
                )
                # additive jitter if you want it (you used gamma for this)
                if gamma and gamma > 0:
                    Zk = Zk + float(gamma) * rngk.normal(size=Zk.shape)
                return Zk
    
            if label_aware:
                y = np.asarray(self.labels)
                uniq = self._stable_unique(y)
                Z = np.empty((n, embed_dim), dtype=float)
                params = {}
    
                # use deterministic per-label RNGs (so label-aware is reproducible)
                base_seed = kwargs.get("poly_seed_base", None)
                for lb in uniq:
                    mask = (y == lb)
                    # optional: per-label seed for reproducibility beyond global rng state
                    if base_seed is None:
                        rngk = self.rng
                    else:
                        rngk = np.random.default_rng(int(base_seed) + int(lb))
    
                    Zk = poly_map(X[mask], rngk)
                    Z[mask] = Zk
                    params[lb] = {"degree": degree}
    
                self._embed_params = {"method": "polynomial", "per_label": params}
                return global_map(Z)
    
            Z = poly_map(X, self.rng)
            return global_map(Z)
    
        raise ValueError("Method should be one of: orthogonal, rff, neural, polynomial")
    
    def _sample_gamma_cluster(self, k, n, rng):
        """
        Sample correlated gamma features using a Gaussian copula.
    
        Cluster-specific dependence comes from self.Cov_k[k].
        Gamma marginals come from self.shape and self.scale[k].
        """
        shape = np.asarray(self.shape, dtype=float)
        if shape.ndim == 0:
            shape = np.full(self.d_0, float(shape))
    
        scale = np.asarray(self.scale[k], dtype=float)
        if scale.shape != (self.d_0,):
            raise ValueError(f"self.scale[{k}] must have shape ({self.d_0},)")
    
        cov = np.asarray(self.Cov_k[k], dtype=float)
        if cov.shape != (self.d_0, self.d_0):
            raise ValueError(f"self.Cov_k[{k}] must have shape ({self.d_0}, {self.d_0})")
    
        # Convert covariance -> correlation
        sd = np.sqrt(np.diag(cov))
        if np.any(sd <= 0):
            raise ValueError("Cov_k must have strictly positive diagonal entries.")
    
        corr = cov / np.outer(sd, sd)
    
        # Numerical stabilization
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
    
        # Sample latent correlated Gaussian
        Z = rng.multivariate_normal(
            mean=np.zeros(self.d_0),
            cov=corr,
            size=n
        )
    
        # Map Gaussian -> Uniform(0,1)
        U = norm.cdf(Z)
    
        # Avoid exact 0/1 for inverse CDF stability
        U = np.clip(U, 1e-12, 1 - 1e-12)
    
        # Map uniforms -> gamma marginals
        X = np.empty((n, self.d_0), dtype=float)
        for j in range(self.d_0):
            X[:, j] = gamma_dist.ppf(U[:, j], a=shape[j], scale=scale[j])
    
        return X
    

def _unit(v):
    n = np.linalg.norm(v)
    return v / (n or 1.0)

def _orthogonal_unit(vhat, rng):
    """
    Return a unit vector orthogonal to vhat in R^d0 (approximately).
    For d0=2 this is exact; for d0>2 it picks a random orthogonal direction.
    """
    d0 = vhat.size
    if d0 == 2:
        w = np.array([-vhat[1], vhat[0]])
        return _unit(w)

    # d0 > 2: pick random w and project out component along vhat
    w = rng.standard_normal(d0)
    w = w - np.dot(w, vhat) * vhat
    return _unit(w)

def _set_pairwise_interlocking(self, sep_frac=0.35, lift_frac=0.50, q=0.95,
                               sep_jitter=0.10, lift_jitter=0.10):
    """
    Pair closest centers, then for each pair:
      - enforce separation ~ sep_frac * r_out along vhat
      - enforce perpendicular offset ~ lift_frac * r_out along what
      - store opposite bite directions u along vhat

    This produces 'S'-style interlocking rather than just facing crescents.
    """
    C = np.asarray(self.cluster_centers, dtype=float).copy()
    K, d0 = C.shape

    pairs, leftover = _pair_centers_greedy(C)

    r_out = float(np.sqrt(chi2.ppf(q, df=self.d_0)))
    base_sep  = float(sep_frac)  * r_out
    base_lift = float(lift_frac) * r_out

    rng = self.rng
    moon_u = {}
    pair_of = {}

    for (i, j) in pairs:
        ci, cj = C[i].copy(), C[j].copy()
        v = cj - ci
        if np.linalg.norm(v) < 1e-12:
            v = np.zeros(d0); v[0] = 1.0

        vhat = _unit(v)
        what = _orthogonal_unit(vhat, rng)

        # jittered magnitudes
        sep  = base_sep  * rng.uniform(1 - sep_jitter,  1 + sep_jitter)
        lift = base_lift * rng.uniform(1 - lift_jitter, 1 + lift_jitter)

        mid = 0.5 * (ci + cj)

        # --- KEY: add perpendicular offset (lift) so moons interlock ---
        C[i] = mid - 0.5 * sep * vhat - 0.5 * lift * what
        C[j] = mid + 0.5 * sep * vhat + 0.5 * lift * what

        # opposite bite directions along vhat
        moon_u[i] = +vhat
        moon_u[j] = -vhat
        pair_of[i] = j
        pair_of[j] = i

    self.cluster_centers = C
    self._moon_u = moon_u
    self._moon_pair_of = pair_of
    self._moon_leftover = leftover

    
