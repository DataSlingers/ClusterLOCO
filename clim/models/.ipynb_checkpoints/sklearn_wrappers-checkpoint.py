"""
Author: Claire He

Sklearn wrappers for alternative clustering algorithms
- Leiden: direct sklearn wrapper on leiden method
- Base Spectral Clustering (based on Luxburg paper)
- Fast Spectral Clustering (with power iterations)
- SpectralClusteringAffinity: supports different affinity transformations
- Gamma Mixture (to double check, generated with GPT)
"""
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA, SparsePCA
import numpy as np
from sklearn.decomposition import TruncatedSVD
import leidenalg 
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import igraph as ig
from scipy.sparse import coo_matrix
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh as geigh
from numpy.linalg import eigh
from scipy.special import gammaln, digamma, polygamma, logsumexp
from sklearn.utils.validation import check_array, check_is_fitted

class Leiden(BaseEstimator, ClusterMixin):
    def __init__(self, n_neighbors=10, resolution=0.6, pc_comp=None):
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.pc_comp = pc_comp

    def fit(self, X, y=None):
        if self.pc_comp is None: 
            A = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
           
        else: 
            V = PCA(n_components=self.pc_comp).fit_transform(X)
            A = kneighbors_graph(V, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
        
        A = A.maximum(A.T)  # symmetrize the graph
        # Convert to igraph
        sources, targets = A.nonzero()
        g = ig.Graph(n=X.shape[0], edges=list(zip(sources, targets)))
        g.vs["name"] = list(map(str, range(X.shape[0])))

        # Run Leiden
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution
        )
        self.labels_ = np.array(partition.membership)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, "resolution": self.resolution}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


    
class BaseSpectralClustering(BaseEstimator, ClusterMixin):
    """ Base Spectral Clustering algorithm: 

    Follows Luxburg's spectral clustering tutorial on Spectral Clustering: https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf

    Supports via normalize=choice('normalize_ng', 'normalize_shi','unnormalized') the 3 typical algorithms for graph laplacian computing and eigensolver.

    """
    def __init__(self, n_clusters=3, similarity_matrix='knn', normalize='normalize_ng', kernel='self_tuning', sym='max', eigen_solver='auto', random_state=0, n_init='auto', n_neighbors=15, **kwargs):
        self.similarity_matrix=similarity_matrix
        self.K = int(n_clusters)
        self.normalize=normalize
        self.kernel = kernel
        self.sym = sym
        self.eigen_solver= eigen_solver
        self.random_state=random_state
        self.n_init=n_init
        self.n_neighbors = n_neighbors

    def _compute_weights_dists(self, dists, rows, cols, X=None, kernel='self_tuning', **kw):
        """
        dists: (nnz,) distances for edges (rows[i], cols[i])
        rows, cols: (nnz,) integer indices
        kernel: 'rbf' | 'self_tuning' | 'cosine'
        """
        if kernel=='rbf':
            gamma = float(kw.get('gamma', 1.0))
            return np.exp(-gamma*(dists**2))
        elif kernel=='self_tuning':
            sigma = kw.get('sigma',None)
            if sigma is None:
                raise ValueError("self_tuning kernel requires sigma=(n,) array.")
            return np.exp(-(dists**2)/(sigma[rows]*sigma[cols] + 1e-12))
        elif kernel=='cosine':
            if X is None:
                raise ValueError("cosine kernel requires X.")
            Xn = X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-12)
            return np.maximum(0, np.sum(Xn[rows]*Xn[cols], axis=1))
        raise ValueError("kernel must be one of:'rbf', 'self_tuning', 'cosine'")

    def _compute_similarity_graph(self, X, **kwargs):
        """
        Returns:
          rows, cols : (nnz,) int arrays describing directed edges i -> j
          dists      : (nnz,) float array with distance d(i,j) for each edge
          meta       : dict with extra per-node quantities (e.g., sigma for self-tuning)
        """
        n = X.shape[0]
        if self.similarity_matrix=='knn':
            # K nearest neighbor graph
            n_neighbors = self.n_neighbors
            n_jobs = kwargs.get("n_jobs", None)
            metric = kwargs.get("metric", "minkowski")
            
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs).fit(X)
            dmat, idx = nn.kneighbors(X)  # (n,k)
            rows, cols, dists = np.repeat(np.arange(n), n_neighbors), idx.ravel(), dmat.ravel()
            sigma = dmat[:, -1]+1e-12 # distance to k-th neighbor
            meta = {"sigma": sigma, "n": n}

            return rows, cols, dists, meta
            
        elif self.similarity_matrix=='epsilon': 
            # Epsilon neighbor graph
            metric = kwargs.get("metric", 'euclidean')
            eps = kwargs.get('epsilon',0.1)
            D = pairwise_distances(X, metric=metric)
            np.fill_diagonal(D, np.inf)
            rows, cols = np.where(D <= eps)
            dists = D[rows, cols]
            sigma = np.full(n, 1.0)
            for i in range(n):
                di = D[i, (D<=eps)[i]]
                if di.size:
                    sigma[i] = np.median(di)
            meta = {"sigma":sigma, "n":n}
            return rows, cols, dists, meta

        elif self.similarity_matrix=='connected':
            # Fully connected
            metric = kwargs.get("metric", 'euclidean')
            D = pairwise_distances(X, metric=metric)
            np.fill_diagonal(D, np.inf)
            meta = {"D":D, "n":n}
            return None, None, None, meta # use dense D in meta
        else: 
            print("Similarity matrix method not supported, choose among 'knn','epsilon','connected'")

    def _compute_weighted_adj(self, X, rows, cols, dists, meta, *, kernel=None, sym=None, sparse_connected=False, **kwargs):
        n = meta["n"]
        kernel=self.kernel if kernel is None else kernel
        sym = self.sym if sym is None else sym
        X = np.asarray(X)
        
        if self.similarity_matrix != "connected":
            sigma=meta.get('sigma', None)
            w = self._compute_weights_dists(dists, rows,cols, kernel=kernel, X=X, sigma=sigma, **kwargs)
            W = sp.coo_matrix((w, (rows, cols)), shape=(n,n)).tocsr()
            if sym=='max':
                W = W.maximum(W.T)
                W.setdiag(0.0)
                W.eliminate_zeros()
                return W
            elif sym=='mean':
                W = 0.5* (W+W.T)
                W.setdiag(0.0)
                W.eliminate_zeros()
                return W  
 
        D = meta["D"] # dense D
        if kernel == "rbf":
            gamma = float(kwargs.get("gamma", 1.0))
            W = np.exp(-gamma * (D ** 2))
            np.fill_diagonal(W, 0.0)

        elif kernel == "self_tuning":
            # use kNN-derived sigma to self-tune even though fully connected
            k = int(kwargs.get("n_neighbors", 15))
            nn = NearestNeighbors(n_neighbors=k, metric=kwargs.get("metric", "euclidean")).fit(X)
            dmat, _ = nn.kneighbors(X)
            sigma = dmat[:, -1] + 1e-12
            W = np.exp(-(D ** 2) / (sigma[:, None] * sigma[None, :] + 1e-12))
            np.fill_diagonal(W, 0.0)

        elif kernel == "cosine":
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            W = np.maximum(0.0, Xn @ Xn.T)
            np.fill_diagonal(W, 0.0)

        else:
            raise ValueError("kernel must be one of: 'rbf', 'self_tuning', 'cosine'")      
        if sym == "max":
            W = np.maximum(W, W.T)
        elif sym == "mean":
            W = 0.5 * (W + W.T)
        else:
            raise ValueError("symmetrize must be 'max' or 'mean'")
        return sp.csr_matrix(W) if sparse_connected else W


    def compute_laplacian(self, X, *,laplacian='sym', **kwargs):
        """
        This now does:
          graph (rows, cols, dists) -> weights -> weighted adjacency W
        """
        rows, cols, dists, meta = self._compute_similarity_graph(X, **kwargs)
        n = meta['n']
        W = self._compute_weighted_adj(X, rows, cols, dists, meta, **kwargs)
        
        if sp.issparse(W):
            D = np.asarray(W.sum(axis=1)).ravel()
        else:
            D = W.sum(axis=1)

        if laplacian == 'unnormalized': # unnormalized laplacian 
            L = sp.diags(D, format='csr')-W if sp.issparse(W) else np.diag(D)-W
            return L, D

        elif laplacian == "sym":
            inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-12)) 
            if sp.issparse(W):
                D_inv_sqrt = sp.diags(inv_sqrt, format="csr")
                L = sp.eye(n, format="csr") - (D_inv_sqrt @ W @ D_inv_sqrt)
            else:
                L = np.eye(n) - (inv_sqrt[:, None] * W * inv_sqrt[None, :])
            return L, D
        
        elif laplacian == "rw":
            inv_D = 1.0 / np.maximum(D, 1e-12)
            if sp.issparse(W):
                D_inv = sp.diags(inv_D, format="csr")
                L = sp.eye(n, format="csr") - (D_inv @ W)
            else:
                L = np.eye(n) - (inv_D[:, None] * W)
            return L, D
        raise ValueError("Laplacian should be 'sym' or 'rw'")
            
    def get_eigenvec_laplacian(self, L, k):
        use_sparse = sp.issparse(L)
        n = L.shape[0]
        if self.eigen_solver=='auto':
            use_sparse = use_sparse or (n > 2000)

        if use_sparse:
            vals, vecs = eigsh(L, k=k, which='SM')
        else:
            vals, vecs = eigh(L) 
            vecs, vals = vecs[:, :k], vals[:k]
        self.evals_ = vals
        U = vecs # (n, k) ui as columns
        self.U_ = U

    def get_geigenvec_laplacian(self, L, D, k):
        use_sparse = sp.issparse(L)
        deg_stable = np.maximum(D, 1e-12)
        if use_sparse:
            Dmat = sp.diags(deg_stable, format='csr')
            vals, U = eigsh(L, k=k, M=Dmat, which='SM')
        else:
            Dmat = np.diag(deg_stable)
            vals, U = geigh(L, Dmat)
            vals, U = vals[:k], U[:, :k]
        self.U_ = U
        self.evals_ = vals
            
            
    def fit(self, X, **kwargs):
        if self.normalize=='normalize_ng': 
            # Compute Laplacian 
            Lsym, _ = self.compute_laplacian(X, laplacian='sym')
            # Get first k eigenvectors 
            self.get_eigenvec_laplacian(Lsym, self.K)
            U = self.U_
            T = U/(np.linalg.norm(U, axis=1, keepdims=True)+1e-12)
            self.embedding_ = T # (row embedding)
        elif self.normalize=='normalize_shi':
            L, D = self.compute_laplacian(X, laplacian='unnormalized')
            self.get_geigenvec_laplacian(L, D, self.K)
            self.embedding_ = self.U_
        else: 
            L, _ = self.compute_laplacian(X, laplacian='unnormalized')
            self.get_eigenvec_laplacian(L, self.K)
            self.embedding_ = self.U_ # (row embedding)
        
        km = KMeans(n_clusters=self.K, n_init=self.n_init, random_state=self.random_state)
        self.labels_ = km.fit_predict(self.embedding_)
        self.clusters_ = [np.flatnonzero(self.labels_ == c) for c in range(self.K)]
        return self

    def fit_predict(self, X, **kwargs):
        return self.fit(X, **kwargs).labels_

    def predict(self, X, **kwargs):
        """ just an alias for fit predict, this is not generative """
        return self.fit_predict(X, **kwargs)

    def get_params(self, deep=True):
        return {"n_clusters": self.K, "n_neighbors": self.n_neighbors, "similarity_matrix":self.similarity_matrix, "normalize":self.normalize}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
        

class FastSpectralClustering(BaseEstimator, ClusterMixin):
    """
    Fast Spectral Clustering based on 
        - Power iterations for graph Laplacian computation: https://jmlr.csail.mit.edu/papers/volume22/20-261/20-261.pdf

    """
    def __init__(self, n_clusters=3, random_state=0):
        sp_base = BaseSpectralClustering(n_clusters=n_clusters, random_state=0)
        self.base_sp = sp_base 
        self.random_state = random_state
        self.K = n_clusters
        
    def _power_method(self, M, X_init, t):
        # X_init, _ = np.linalg.qr(X_init, mode="reduced")
        X = X_init
        for i in range(t):
            X = M @ X
        return X

    def fit(self, X, **kwargs):
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        L, D = self.base_sp.compute_laplacian(X, laplacian='sym', **kwargs)
        if sp.issparse(L):
            I = sp.eye(n, format="csr")
        else:
            I = np.eye(n)
        M = I - 0.5*L
        l = kwargs.get("l", max(int(np.ceil(np.log(self.K))), 1))
        t = kwargs.get("t", max(int(np.ceil(10 * np.log(n/self.K))),1))
        Y = np.empty((n, l))
        for it in range(l):
            x0 = rng.normal(size=(n,))
            Y[:,it] = self._power_method(M, x0, t)
        
        inv_sqrt_D = 1.0 / np.sqrt(np.maximum(D, 1e-12))
        self.km = KMeans(n_clusters=self.K, random_state=self.random_state)
        Z = inv_sqrt_D[:, None] * Y
        self.embedding_ = Z # / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        self.labels_ = self.km.fit_predict(self.embedding_)
        self.clusters_ = [np.flatnonzero(self.labels_ == c) for c in range(self.K)]

        return self

    def get_params(self, deep=True):
        return {"n_clusters": self.K}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit_predict(self, X, **kwargs):
        return self.fit(X, **kwargs).labels_

    def predict(self, X):
        """ just an alias for fit_predict, not generative """
        return self.fit_predict(X)
        
class SpectralClusteringAffinity(BaseEstimator, ClusterMixin):
    """
    Modified Spectral Clustering that supports different methods to compute affinity matrix. 
    Supports PCA reduction on affinity computation for high dimensional data when affinity name trails "_pca". 
    Spectral Clustering algorithm backbone follows sklearn.cluster's SpectralClustering passing affinity as 'precomputed'.

    Parameters
    ---------
    alpha: default (int) | 10 (0 for regular PCA, otherwise uses SparsePCA with alpha regularizer)
    affinity: default (str) | 'self': 'heat', 'self','binary' and 'heat_pca','self_pca','binary_pca'
        'heat': uses a heat kernel on kNN affinity matrix
        'self': uses a gaussian kernel kNN affinity matrix 
        'binary': uses kNN affinity matrix as is
    """
    def __init__(self,n_clusters=2, n_components=3, alpha=10, n_neighbors=4, affinity='self', random_state=42):
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.n_components = n_components
        self.affinity = affinity
        self.random_state = random_state

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        # Clip PCA dimension to what is actually possible on this minipatch
        n_comp_eff = min(self.n_components, n_samples - 1, n_features)
        # Be safe in corner cases
        n_comp_eff = max(1, n_comp_eff)

        
        if self.alpha == 0:
            pca = PCA(n_components=n_comp_eff, random_state=self.random_state) 
        else:
            pca = SparsePCA(n_components=n_comp_eff, alpha=self.alpha, random_state=self.random_state)
        Z = pca.fit_transform(X)
        
        if self.affinity == "binary_pca":
            G = kneighbors_graph(Z, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
            A = G.maximum(G.T)

        elif self.affinity == "heat_pca":
            D = kneighbors_graph(Z, n_neighbors=self.n_neighbors, mode='distance', include_self=False)
            sigma = np.median(D.data) + 1e-12
            W = D.copy(); W.data = np.exp(-(W.data ** 2) / (2 * sigma ** 2))
            A = W.maximum(W.T)

        elif self.affinity == "self_pca":
            nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(Z)
            dists, idx = nn.kneighbors(Z)
            sigma_i = dists[:, -1] + 1e-12
            rows = np.repeat(np.arange(X.shape[0]), self.n_neighbors)
            cols = idx.ravel()
            vals = np.exp(-(dists.ravel()**2) / (sigma_i[rows] * sigma_i[cols]))
            A = coo_matrix((vals, (rows, cols)), shape=(Z.shape[0], Z.shape[0])).tocsr()
            A = A.maximum(A.T)

        elif self.affinity == "binary":
            G = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
            A = G.maximum(G.T)
       
        elif self.affinity == "heat":
            D = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='distance', include_self=False)
            sigma = np.median(D.data) + 1e-12
            W = D.copy(); W.data = np.exp(-(W.data ** 2) / (2 * sigma ** 2))
            A = W.maximum(W.T)

        elif self.affinity == "self":
            nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
            dists, idx = nn.kneighbors(X)
            sigma_i = dists[:, -1] + 1e-12
            rows = np.repeat(np.arange(Z.shape[0]), self.n_neighbors)
            cols = idx.ravel()
            vals = np.exp(-(dists.ravel()**2) / (sigma_i[rows] * sigma_i[cols]))
            A = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0])).tocsr()
            A = A.maximum(A.T)
                               
        self.A = A
        self.model = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed',
                                               assign_labels='kmeans', random_state=self.random_state)
        
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.model.fit_predict(self.A)

    def predict(self, X, y=None):
        return self.fit_predict(X)

    def get_params(self, deep=True):
        return {"n_clusters": self.n_clusters, "n_components": self.n_components, "alpha": self.alpha, "n_neighbors": self.n_neighbors, "affinity":self.affinity}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self



class GammaMixture(BaseEstimator, ClusterMixin):
    """
    Mixture of independent Gamma distributions fit by EM.

    For component k and feature j:
        X_j | Z=k ~ Gamma(shape=alpha_{k,j}, scale=theta_{k,j})

    Assumes features are independent within each component.

    Parameters
    ----------
    n_components : int, default=3
        Number of mixture components.

    max_iter : int, default=200
        Maximum number of EM iterations.

    tol : float, default=1e-4
        Convergence tolerance on log-likelihood.

    reg_covar : float, default=1e-6
        Small positive value for numerical stability.

    n_init : int, default=1
        Number of random initializations.

    init_params : {"kmeans", "random"}, default="kmeans"
        Initialization method for responsibilities.

    random_state : int or None, default=None
        Random seed.

    verbose : bool, default=False
        Whether to print EM progress.
    """

    def __init__(
        self,
        n_components=3,
        max_iter=200,
        tol=1e-4,
        reg_covar=1e-6,
        n_init=1,
        init_params="kmeans",
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose

    def _check_X(self, X):
        X = check_array(X, dtype=float, ensure_2d=True)
        if np.any(X <= 0):
            raise ValueError("GammaMixture requires strictly positive features.")
        return X

    def _initialize_resp(self, X, rng):
        n, _ = X.shape
        K = self.n_components

        if self.init_params == "kmeans":
            labels = KMeans(
                n_clusters=K,
                n_init=10,
                random_state=rng.integers(1_000_000_000),
            ).fit_predict(X)
            resp = np.zeros((n, K))
            resp[np.arange(n), labels] = 1.0
        elif self.init_params == "random":
            resp = rng.uniform(size=(n, K))
            resp /= resp.sum(axis=1, keepdims=True)
        else:
            raise ValueError("init_params must be 'kmeans' or 'random'.")

        return resp

    @staticmethod
    def _weighted_mean(x, w):
        return np.sum(w * x) / np.sum(w)

    @staticmethod
    def _weighted_mean_log(x, w):
        return np.sum(w * np.log(x)) / np.sum(w)

    def _solve_alpha(self, x, w, alpha_init):
        """
        Solve for Gamma shape alpha under weighted MLE:
            log(alpha) - psi(alpha) = log(mean_x) - mean(log x)
        using Newton iterations.
        """
        mean_x = self._weighted_mean(x, w)
        mean_log_x = self._weighted_mean_log(x, w)
        c = np.log(mean_x) - mean_log_x

        alpha = max(alpha_init, 1e-3)
        for _ in range(100):
            f = np.log(alpha) - digamma(alpha) - c
            fp = 1.0 / alpha - polygamma(1, alpha)
            step = f / fp
            alpha_new = alpha - step
            if alpha_new <= 0 or not np.isfinite(alpha_new):
                alpha_new = alpha / 2
            if abs(alpha_new - alpha) < 1e-8:
                return max(alpha_new, 1e-6)
            alpha = alpha_new

        return max(alpha, 1e-6)

    def _estimate_gamma_params(self, X, resp):
        n, d = X.shape
        K = self.n_components

        nk = resp.sum(axis=0) + 10 * np.finfo(float).eps
        weights = nk / n

        alphas = np.zeros((K, d))
        thetas = np.zeros((K, d))

        for k in range(K):
            w = resp[:, k] + 1e-16

            for j in range(d):
                x = X[:, j]
                mean_x = self._weighted_mean(x, w)
                var_x = np.sum(w * (x - mean_x) ** 2) / np.sum(w)
                var_x = max(var_x, self.reg_covar)

                # Method-of-moments init
                alpha_init = max(mean_x**2 / var_x, 1e-3)

                # Weighted MLE refinement
                alpha = self._solve_alpha(x, w, alpha_init)
                theta = mean_x / alpha

                alphas[k, j] = max(alpha, 1e-6)
                thetas[k, j] = max(theta, 1e-6)

        return weights, alphas, thetas

    def _estimate_log_prob(self, X, alphas, thetas):
        """
        Log p(X | Z=k) for each sample and component.
        Independence across features.
        """
        n, d = X.shape
        K = alphas.shape[0]
        log_prob = np.zeros((n, K))

        logX = np.log(X)

        for k in range(K):
            a = alphas[k]       # (d,)
            t = thetas[k]       # (d,)

            # Sum over features of Gamma log-density
            # (a-1)log x - x/t - a log t - log Gamma(a)
            lp = (
                (a - 1.0) * logX
                - X / t
                - a * np.log(t)
                - gammaln(a)
            ).sum(axis=1)

            log_prob[:, k] = lp

        return log_prob

    def _e_step(self, X, weights, alphas, thetas):
        log_prob = self._estimate_log_prob(X, alphas, thetas)
        log_prob_weighted = log_prob + np.log(weights + 1e-16)

        log_norm = logsumexp(log_prob_weighted, axis=1)
        log_resp = log_prob_weighted - log_norm[:, None]
        resp = np.exp(log_resp)

        return resp, log_norm.sum()

    def _m_step(self, X, resp):
        return self._estimate_gamma_params(X, resp)

    def _fit_once(self, X, rng):
        resp = self._initialize_resp(X, rng)
        weights, alphas, thetas = self._m_step(X, resp)

        lower_bound = -np.inf

        for it in range(self.max_iter):
            resp, log_lik = self._e_step(X, weights, alphas, thetas)
            weights, alphas, thetas = self._m_step(X, resp)

            change = log_lik - lower_bound
            if self.verbose:
                print(f"iter={it:3d} loglik={log_lik:.6f} change={change:.6f}")

            if abs(change) < self.tol:
                break

            lower_bound = log_lik

        return {
            "weights": weights,
            "alphas": alphas,
            "thetas": thetas,
            "resp": resp,
            "lower_bound": lower_bound,
            "n_iter": it + 1,
        }

    def fit(self, X, y=None):
        X = self._check_X(X)
        rng = np.random.default_rng(self.random_state)

        best = None
        best_lb = -np.inf

        for init_idx in range(self.n_init):
            run_rng = np.random.default_rng(rng.integers(1_000_000_000))
            result = self._fit_once(X, run_rng)

            if result["lower_bound"] > best_lb:
                best = result
                best_lb = result["lower_bound"]

        self.weights_ = best["weights"]
        self.alphas_ = best["alphas"]
        self.thetas_ = best["thetas"]
        self.resp_ = best["resp"]
        self.labels_ = np.argmax(best["resp"], axis=1)
        self.lower_bound_ = best["lower_bound"]
        self.n_iter_ = best["n_iter"]
        self.n_features_in_ = X.shape[1]

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["weights_", "alphas_", "thetas_"])
        X = self._check_X(X)
        resp, _ = self._e_step(X, self.weights_, self.alphas_, self.thetas_)
        return resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def score_samples(self, X):
        check_is_fitted(self, ["weights_", "alphas_", "thetas_"])
        X = self._check_X(X)
        log_prob = self._estimate_log_prob(X, self.alphas_, self.thetas_)
        return logsumexp(log_prob + np.log(self.weights_ + 1e-16), axis=1)

    def score(self, X, y=None):
        return np.mean(self.score_samples(X))