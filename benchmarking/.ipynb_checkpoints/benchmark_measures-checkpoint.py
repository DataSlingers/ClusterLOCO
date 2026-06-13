"""
Cluster feature importance metrics in the literature for benchmarking.

Napoles 2024:
    - PBFI: for any method well described by prototypes 
        helper: scoring_prototypes 
    - c_SHAP: uses Fuzzy-kmeans as surrogates,
        helper: 
            - SHAP_fuzzy_per_point:  SHAP sampling scheme using permutation sampling for each point
            - compute_shapley_membership: computes exact SHAP
Montavon 2021:
    - LRP_score: uses neuralized k-means to compute layer-wise relevant propagation scores.
    - LRP_cluster: computes cluster-level aggregated LRP scores.

Pfaffel (R vignette):
    - Permutation feature importance score https://github.com/o1iv3r/FeatureImpCluster/blob/master/R/

LOCO-style score:
    - LOCO_silhouette: LOCO score with Silhouette score instead 
    - GlobalStability: Ben-Hur stability LOCO 

Author: Claire HE 
"""
import numpy as np
from benchmarking.neuralized_kmeans import *
from benchmarking.prototypes import *
from sklearn.base import clone
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import scipy.special as sc
from sklearn.linear_model import LinearRegression
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, silhouette_score,accuracy_score,confusion_matrix
from math import comb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import tqdm
from sklearn.neighbors import kneighbors_graph
import sys
sys.path.append("../")
from clim.utils.model_selection import *
from clim.utils.utils import *



class Fuzzy_CSHAP_explainer:
    def __init__(self, X, K, model=None, baseline="mean", X_reference=None, random_state=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        self.X = X
        self.K = K
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if model is None:
            model = FuzzyKMeans(n_clusters=K)
        self.model = model
        self.model.fit(X)

        self.centroids = np.asarray(self.model.cluster_centers_, dtype=float)
        self.m = float(self.model.m)
        self.n_features = self.centroids.shape[1]

        if self.m <= 1:
            raise ValueError("Fuzzifier m must be > 1.")

        if baseline == "mean":
            ref = X if X_reference is None else np.asarray(X_reference, dtype=float)
            self.baseline_vec = ref.mean(axis=0)
        elif baseline == "zero":
            self.baseline_vec = np.zeros(self.n_features, dtype=float)
        else:
            raise ValueError("baseline must be 'mean' or 'zero'.")

    def _get_membership_vector(self, x):
        """
        Return fuzzy memberships for all clusters for a single point x.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"x must have length {self.n_features}.")

        dists = np.linalg.norm(self.centroids - x, axis=1)

        # Handle exact matches to one or more centroids
        zero_mask = np.isclose(dists, 0.0)
        if np.any(zero_mask):
            memberships = np.zeros(self.K, dtype=float)
            memberships[zero_mask] = 1.0 / zero_mask.sum()
            return memberships

        power = 2.0 / (self.m - 1.0)
        inv_dists = 1.0 / (dists ** power)
        memberships = inv_dists / inv_dists.sum()
        return memberships

    def _get_membership(self, x, cluster_idx):
        """
        Return membership for one cluster for a single point x.
        """
        if not (0 <= cluster_idx < self.K):
            raise IndexError(f"cluster_idx must be in [0, {self.K - 1}].")
        return self._get_membership_vector(x)[cluster_idx]

    def explain(self, x, cluster_idx, method="kernel", M=100):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"x must have length {self.n_features}.")

        if method == "exact":
            return self._exact_shapley(x, cluster_idx)
        elif method == "kernel":
            return self._kernel_shap(x, cluster_idx, M)
        elif method == "perm":
            return self._permutation_shap(x, cluster_idx, M)
        else:
            raise ValueError("method must be 'exact', 'kernel', or 'perm'.")

    def _permutation_shap(self, x, cluster_idx, M):
        """
        Permutation SHAP estimator for a fixed fitted fuzzy clustering model.
        Returns a vector of feature attributions.
        """
        if M <= 0:
            raise ValueError("M must be a positive integer.")

        phi = np.zeros(self.n_features, dtype=float)

        for _ in range(M):
            perm = self.rng.permutation(self.n_features)
            x_curr = self.baseline_vec.copy()
            prev_mu = self._get_membership(x_curr, cluster_idx)

            for i in perm:
                x_curr[i] = x[i]
                curr_mu = self._get_membership(x_curr, cluster_idx)
                phi[i] += (curr_mu - prev_mu)
                prev_mu = curr_mu

        return phi / M

    def _exact_shapley(self, x, cluster_idx):
        """
        Exact Shapley values for the fixed-model membership function.
        Only practical for small numbers of features.
        """
        phi = np.zeros(self.n_features, dtype=float)
        all_features = list(range(self.n_features))

        for i in range(self.n_features):
            others = [f for f in all_features if f != i]

            for k in range(len(others) + 1):
                for subset in combinations(others, k):
                    subset = list(subset)

                    idx_with = np.array(subset + [i], dtype=int)
                    idx_without = np.array(subset, dtype=int)

                    x_with = self.baseline_vec.copy()
                    x_with[idx_with] = x[idx_with]

                    x_without = self.baseline_vec.copy()
                    if len(idx_without) > 0:
                        x_without[idx_without] = x[idx_without]

                    v_with = self._get_membership(x_with, cluster_idx)
                    v_without = self._get_membership(x_without, cluster_idx)

                    weight = (
                        sc.factorial(len(subset))
                        * sc.factorial(self.n_features - len(subset) - 1)
                        / sc.factorial(self.n_features)
                    )
                    phi[i] += weight * (v_with - v_without)

        return phi

    def _kernel_shap(self, x, cluster_idx, M):
        """
        KernelSHAP-style weighted linear regression approximation.
        Returns (phi0, phi), where phi0 is the intercept/base value.
        """
        if M < 2:
            raise ValueError("M must be at least 2 for kernel SHAP.")

        masks = [
            np.zeros(self.n_features, dtype=int),
            np.ones(self.n_features, dtype=int),
        ]

        while len(masks) < M:
            mask = self.rng.integers(0, 2, size=self.n_features)
            s = mask.sum()
            if 0 < s < self.n_features:
                masks.append(mask)

        X_masks = np.asarray(masks, dtype=float)
        y = np.zeros(len(X_masks), dtype=float)
        weights = np.zeros(len(X_masks), dtype=float)

        for j, mask in enumerate(X_masks):
            s = int(mask.sum())
            x_eval = np.where(mask == 1, x, self.baseline_vec)
            y[j] = self._get_membership(x_eval, cluster_idx)

            if s == 0 or s == self.n_features:
                weights[j] = 1e6
            else:
                weights[j] = (
                    (self.n_features - 1)
                    / (sc.comb(self.n_features, s) * s * (self.n_features - s))
                )

        reg = LinearRegression(fit_intercept=True)
        reg.fit(X_masks, y, sample_weight=weights)

        phi0 = float(reg.intercept_)
        phi = np.asarray(reg.coef_, dtype=float)
        return phi0, phi

    def hard_labels(self, X):
        """
        Hard cluster assignment from max membership under the fixed fitted model.
        """
        X = np.asarray(X, dtype=float)
        labels = np.array([np.argmax(self._get_membership_vector(x)) for x in X], dtype=int)
        return labels

def _local_phi_worker(x, explainer, cluster_idx, method, M):
    result = explainer.explain(x, cluster_idx, method=method, M=M)
    phi = result[1] if method == "kernel" else result
    return np.abs(phi)
    
class c_SHAP:
    def __init__(self, X, K, model=None, method='perm', M=100, n_jobs=8,
                 X_reference=None, random_state=None, baseline='zero'):
        self.X = np.asarray(X, dtype=float)
        self.K = K
        self.method = method
        self.M = M
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.explainer = Fuzzy_CSHAP_explainer(
            X=self.X,
            K=self.K,
            model=model,
            baseline=baseline,
            X_reference=X_reference,
            random_state=self.random_state
        )
        self.model = self.explainer.model

    def get_global_importance(self, cluster_idx, X_subset=None):
        X_eval = np.asarray(self.X if X_subset is None else X_subset, dtype=float)
    
        abs_shaps = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_local_phi_worker)(x, self.explainer, cluster_idx, self.method, self.M)
            for x in X_eval
        )
        return np.mean(abs_shaps, axis=0)

    def get_model_wide_importance(self):
        """
        Compute one global importance score per feature across the whole fitted model.
        Uses hard cluster assignments and averages cluster-wise importances weighted by cluster size.
        """
        X = self.X
        total_importance = np.zeros(X.shape[1], dtype=float)

        labels = self.explainer.hard_labels(X)

        for j in range(self.K):
            cluster_points = X[labels == j]
            if len(cluster_points) == 0:
                continue

            cluster_phi = self.get_global_importance(
                cluster_idx=j,
                X_subset=cluster_points
            )

            total_importance += cluster_phi * (len(cluster_points) / len(X))

        return total_importance, labels

def scoring_prototypes(z, i, norm='l1'):
    """ Pairwise distance of prototypes to prototype i
    
    Parameters
    ----------
    z: ndarray, shape (n, )
        fitted labels 
    i: int
        index of cluster to compute pairwise distance to
    norm: str
        uses either l1 or l2 norm, 'l1' or 'l2'
        
    Returns
    ----------
    scoring of l1 pairwise or l2 pairwise distance for prototype of cluster i
    """
    if norm=='l1':
        apply_func = lambda x: np.abs(x)
    elif norm=='l2':
        apply_func = lambda x: x**2
    K = z.shape[0]
    phi_i = 0
    for j in range(K):
        for l in range(K):
            phi_i += apply_func(z[j, i]- z[l, i])
    return phi_i


def PBFI(X, model, K=None, centroid='means'):
    """
    From Napoles, prototype-based feature importance for clustering.

    Parameters
    ----------
    X: ndarray, shape (n, d)
        data
    model: 
        sklearn type model, forward with fit_predict, if model does not have cluster_centers_, need to specify centroid choice.
    K: int
        number of clusters

    Returns 
    ----------
    PBFI score, shape (d, )
    """
    if K is not None: # model not instantiated or needing reinstantiation
        model = clone(model)
        model.set_params(n_clusters=K)
    elif K is None:
        K = model.n_clusters
    z = model.fit_predict(X)
    if hasattr(model, "cluster_centers_"):
        cluster_center = np.array([model.cluster_centers_[i] for i in range(K)]) # (K, d)
    else:
        if centroid=='means':
            cluster_center = np.array([X[z == k, :].mean(axis=0) for k in range(K)])
        elif centroid == 'median':
            cluster_center = np.array([X[z == k, :].median(axis=0) for k in range(K)])
    pbfi = np.zeros(X.shape[1], )
    for d in range(X.shape[1]): 
        pbfi[d] = scoring_prototypes(cluster_center, d)
    return pbfi/sum(pbfi)


def LRP_score(X, K, random_state=42):
    """ 
    Based on Montavon, Kauffmann neuralized K-means (NEON)
    Needs to use a Kmeans as surrogate.

    Parameters
    ----------
    X: ndarray, shape (n, d)
        data
    K: int
        number of clusters to fit
    random_state: int
        for reproducibility
        
    Returns
    ----------
    Feature-averaged relevance scores, shape (d, )
    """
    X = MinMaxScaler().fit_transform(X) # pass through a neural network need to make sure nothing blows up
    model = KMeans(n_clusters=K, random_state=random_state)
    model.fit(X)
    X_tensor = torch.from_numpy(X)
    logits = margins_kmeans(X_tensor, model)
    nm = NeuralizedKMeans(model)
    R = neon(nm, X_tensor, beta=1.0)
    return R.numpy().mean(axis=0)



def perm_misclass_rate(cluster_obj, X, var_name, base_pred=None, pred_fn = None, biter=5, seed=123):
    base_pred = new_pred = None

    if pred_fn is None:
        if hasattr(cluster_obj, "predict"):
            pred_fn = lambda obj, X_new: obj.predict(X_new)
        else:
            raise ValueError("Provide pred_fn or cluster_obj must have .predict")

    n = X.shape[0]

    # Baseline : cluster before permutation
    if base_pred is None: 
        base_labels = np.asarray(pred_fn(cluster_obj, X), dtype=int)
    else:
        if len(base_pred) != n:
            raise ValueError("Length of base predictions must equal number of observations in the data")
        base_labels = np.asarray(base_pred, dtype=int)

    rng = np.random.default_rng(seed)
    mcr = np.zeros(biter)

    for b in range(biter):
        X_perm = X.copy()
        current_base = base_labels

        perm_idx = rng.permutation(n) # shuffle index for column to break relationship with
        X_perm[:, var_name] = X_perm[perm_idx, var_name]
        # Predict after permutating variable
        new_pred = np.asarray(pred_fn(cluster_obj, X_perm), dtype=int)
        mcr[b] = np.mean(new_pred != current_base)

    return mcr


from numpy.random import SeedSequence 

def feature_imp_cluster(cluster_obj, X, base_pred = None, pred_fn =None, biter=10, seed=123):
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        vars_ = list(X.columns)
        use_names = True
        X_master = X
    else:
        X_master = np.asarray(X)
        vars_ = [f"x{j}" for j in range(X_master.shape[1])]
        use_names = False

    # baseline predictions once (saves time)
    if pred_fn is None:
        if hasattr(cluster_obj, "predict"):
            pred_fn_use = lambda obj, newdata: obj.predict(newdata)
        else:
            raise ValueError("Provide pred_fn or use a cluster_obj with a .predict method")
    else:
        pred_fn_use = pred_fn
            
    if base_pred is None:
        base_pred = np.asarray(pred_fn_use(cluster_obj, X_master), dtype=int)
    else:
        base_pred = np.asarray(base_pred, dtype=int)

    ss = SeedSequence(seed)
    child_seeds = ss.spawn(len(vars_))

    # Compute misClassRate Matrix
    mis_mat = np.zeros((biter, len(vars_)), dtype=float)

    for j, name in enumerate(vars_):
        # We call our R-style perm_misclass_rate
        # Note: We pass the clean X_master; the sub-function handles copying.
        mis = perm_misclass_rate(
            cluster_obj=cluster_obj,
            X=X_master,
            var_name=name if isinstance(X, pd.DataFrame) else j,
            base_pred=base_pred,
            pred_fn=pred_fn_use,
            biter=biter,
            seed=child_seeds[j],  
        )
        mis_mat[:, j] = mis

    mis_df = pd.DataFrame(mis_mat, columns=vars_)
    feat_imp = mis_df.mean(axis=0) 

    return {
        "misClassRate": mis_df,
        "featureImp": feat_imp,
        "iterations": biter,
        "seed": seed,
    }   


def LOCO_silhouette(model, X, diff=True):
    """ 
    Computes leave one covariate out for silhouette score
    
    Parameters
    ----------
    model: sklearn-type cluster algorithm
        initialized model, model forward logic must be rewritten with .fit_predict() (sklearn)
    X: ndarray, shape (n_samples, n_features)
        Data 
    diff: Boolean
        if return diff or full scores
    Returns
    -------
    S: full score then LOCO silhouette index score per feature, shape (p+1, )
    z: LOCO cluster assignment per feature, shape (n, p)
    
    """
    S = []
    n, p = X.shape
    z = np.empty((n, p))
    z_full = model.fit_predict(X)
    S.append(silhouette_score(X, z_full))
    for j in range(p):
        X_j = np.delete(X, j, axis=1)
        z[:, j] = model.fit_predict(X_j)
        S.append(silhouette_score(X, z[:,j]))
    if diff:
        return [S[0] - S[j] for j in range(1, len(S))]
    else:
        return S, z



class GlobalStability():
    def __init__(self, X, alg, n_clusters, seed = 234, clf='default', **params):
        self.X = X
        self.model = alg
        self.n_clusters = n_clusters
        self.seed = seed 
        self.params = params
        self.clf = clf
    
    def _modelX_per_split_(self, X1, X2, idx1, idx2, model, j = None, random_state = None):
        if j is not None:
            X1 = np.delete(X1, j, axis=1)
            X2 = np.delete(X2, j, axis=1)
            
        # Ensure we have indices and at least some overlap
        if idx1 is None or idx2 is None or len(idx1) < 1 or len(idx2) < 1:
            pass
        shared, map1, map2 = np.intersect1d(idx1, idx2, return_indices=True)
        if shared.size == 0:
            # no common points → skip this split (ARI undefined on disjoint sets)
            print(f'{i}th split has no intersection, skipping')
            pass
        try:
            model1 = model.set_params(random_state=random_state+1)
            model1 = clone(model1)
            model2 = model.set_params(random_state=random_state-1)
            model2 = clone(model2)
        except ValueError:
            model1 = clone(model)
            model2 = clone(model)

        # Update parameters if needed
        # If connectivity matrix, build one per subset
        if 'n_neighbors'  in self.params :
            for model, Xsub in ((model1, X1), (model2, X2)):
                model = _add_connectivity_(model, Xsub, n_neighbors=n_neighbors)
        
        # Fit/predict
        try:
            y1 = model1.fit_predict(X1)
        except AttributeError:
            model1.fit(X1)
            y1 = getattr(model1, "labels_", None)
            if y1 is None:
                y1 = model1.predict(X1)

        try:
            y2 = model2.fit_predict(X2)
        except AttributeError:
            model2.fit(X2)
            y2 = getattr(model2, "labels_", None)
            if y2 is None:
                y2 = model2.predict(X2)

        # Compare only on intersection of original indices
        shared, idx1_map, idx2_map = np.intersect1d(idx1, idx2, return_indices=True)

        ari = adjusted_rand_score(y1[idx1_map], y2[idx2_map])
        return ari 

    def _consensus_per_split(self, model, X1, idx1, X2, idx2, ref_labels, j=None, seed=None, tau=0.1):
        N = self.X.shape[0]
        M, W = np.zeros((N, N)), np.zeros((N,N))
        O, idx1_map, idx2_map = np.intersect1d(idx1, idx2, return_indices=True)
        if O.size == 0:
            print('No intersecting labels, skip iteration')
        try:
            model1 = model.set_params(random_state=seed+1)
            model2 = model.set_params(random_state=seed-1)
        except ValueError:
            model1 = clone(model)
            model2 = clone(model)
            
        if 'n_neighbors'  in self.params :
            for model, Xsub in ((model1, X1), (model2, X2)):
                model = _add_connectivity_(model, Xsub, n_neighbors=n_neighbors)
        if j is None: 
            C1 = model1.fit_predict(X1)
            C2 = model2.fit_predict(X2)

            # align to reference 
            C1 = match_labels(ref_labels[O], C1[idx1_map])
            C2 = match_labels(ref_labels[O], C2[idx2_map])
            
            M, W = self._consensus_update(M, W, C1, C2, O)
        else: 
            X1_j, X2_j = np.delete(X1, j, axis=1), np.delete(X2, j, axis=1)
            C1 = model1.fit_predict(X1_j)
            C2 = model2.fit_predict(X2_j)

            # align to reference 
            C1 = match_labels(ref_labels[O], C1[idx1_map])
            C2 = match_labels(ref_labels[O], C2[idx2_map])

            M, W = self._consensus_update(M, W, C1, C2, O)

        # Normalize scores
        M_hat = np.zeros_like(M, dtype=float)
        mask = (W > 0)
        M_hat[mask] = M[mask]/W[mask]

        score = self._pac_score(M_hat, tau=tau)
        return score 
            
    def _consensus_update(self, M, W, C1, C2, O):
        # Update M and W for all i, j in O (in-place)
        # From the same consensus definition from Kai's project
        idx = np.ix_(O, O)
        # M[np.ix_(O,O)] = np.asarray([[1*(C1[i] == C1[j]) for i in range(len(O))] for j in range(len(O))])
        C1_eq = (C1[:, None] == C1[None, :]).astype(M.dtype)
        C2_eq = (C2[:, None] == C2[None, :]).astype(M.dtype)
        M[idx] += C1_eq + C2_eq
        W[idx] += 2                # W_{f,k} += 2
        return M, W

    def _pac_score(self, M, tau=0.1):
        tri = np.triu_indices(M.shape[0], k=1)
        vals = M[tri]
        tri_valid = np.ones_like(vals, dtype=bool)
        finite = np.isfinite(vals)
        denom = np.count_nonzero(tri_valid & finite)
        if denom==0:
            return np.nan
        numer = np.count_nonzero((vals > tau) & (vals < (1-tau)))
        return numer/denom
        
        
    def stability(self, B=10, ratio=0.6, method='model-explorer'):
        X = self.X
        model = self.model 
        K = self.n_clusters
        seed = self.seed
        N, d = X.shape

        if K is not None:
            try:
                model.set_params(n_clusters=K)
            except ValueError:
                # Estimator doesn't have n_clusters: ignore
                pass

        scores = np.zeros((B, d))
        if method == 'model-explorer':
            for i in range(B):
                X1, idx1 = data_split(X, method='subsample', ratio=ratio, shuffle=True, ind=True, random_state=seed+i)
                X2, idx2 = data_split(X, method='subsample', ratio=ratio, shuffle=True, ind=True, random_state=seed-i)
                ari = self._modelX_per_split_(X1, X2, idx1, idx2, model, j = None, random_state = seed)
                for j in range(d):
                    ari_j = self._modelX_per_split_(X1, X2, idx1, idx2, model, j = j, random_state = seed)
                    scores[i, j] = ari - ari_j
            self.scores = scores

        elif method == 'consensus':
            # Get reference clustering on all data
            C0 = model.fit_predict(X)
            for i in range(B):
                X1, idx1 = data_split(X, method='subsample', ratio=ratio, shuffle=True, ind=True, random_state=seed+i)
                X2, idx2 = data_split(X, method='subsample', ratio=ratio, shuffle=True, ind=True, random_state=seed-i)
                pac_score = self._consensus_per_split(model, X1, idx1, X2, idx2, C0, seed=seed+i)
                for j in range(d):
                    pac_score_j = self._consensus_per_split(model, X1, idx1, X2, idx2, C0, j=j, seed=seed+i)
                    scores[i, j] = -(pac_score - pac_score_j)
                    
        return np.mean(scores, axis=0)

    def _add_connectivity_(model, X, connectivity_builder=None, n_neighbors=10):
        if "connectivity" in model.get_params():
            if connectivity_builder is not None:
                conn = connectivity_builder(X)
            else:
                # safe default kNN graph on the *subset*
                k_use = min(n_neighbors, X.shape[0] - 1)
                if k_use >= 1:
                    C = kneighbors_graph(X, n_neighbors=k_use, include_self=False)
                    conn = 0.5 * (C + C.T)
                else:
                    conn = None
            if conn is not None:
                model.set_params(connectivity=conn)
        return model
    
    def exact_shapley(self, X_train, X_test, model, value_function, baseline_value=0.0, use_cache=True):
        X_train = np.asarray(X_train)
        X_test  = np.asarray(X_test)
        _, M = X_train.shape
        feats = tuple(range(M))
        cache = {} if use_cache else None
    
        def stable_seed(subset):
            # deterministic seed per subset, per object seed
            return (hash(tuple(sorted(subset))) ^ int(self.seed)) & 0x7fffffff
    
        def v_of(subset):
            key = tuple(sorted(subset))
            if len(key) == 0:
                return baseline_value
            if cache is not None and key in cache:
                return cache[key]
    
            rs = stable_seed(key)
    
            m = clone(model) if model is not None else None
            if m is not None and hasattr(m, "set_params"):
                try:
                    m.set_params(random_state=rs)
                except Exception:
                    pass
    
            # IMPORTANT: pass random_state through so RF is stable too
            val = float(value_function(X_train[:, key], X_test[:, key], m, random_state=rs))
            if cache is not None:
                cache[key] = val
            return val
    
        phi = np.zeros(M, dtype=float)
        for j in feats:
            others = [i for i in feats if i != j]
            contrib = 0.0
            for s in range(0, M):  # 0..M-1 is fine
                w = (1.0 / M) * (1.0 / comb(M - 1, s))
                for S in combinations(others, s):
                    contrib += w * (v_of(S + (j,)) - v_of(S))
            phi[j] = contrib
        return phi
         

from types import SimpleNamespace

def _metric_wants_proba(metric, y_true, K):
    """
    Return True if metric can be called as metric(y_true, probs) and says proba=True.
    """
    if metric == 'ARI':
        return False
    if not callable(metric):
        return False
    try:
        n = len(y_true)
        if n == 0: 
            return False
        dummy = np.full(n, int(np.asarray(y_true)[0]), dtype=int)
        probs = np.full((len(y_true), K), 1.0 / K, dtype=float)
        out = metric(dummy, probs)
        return bool(getattr(out, "proba", False))
    except Exception:
        return False

