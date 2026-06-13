"""" Author: Claire He



Global feature importance score for clustering: Minipatched Stability scores

- based on Ben-Hur stability 
- based on consensus-clustering derived stability (IMPACC based on Gan et al.)

"""
from clim.minipatches.minipatch import get_minipatch, adaptive_minipatch
from clim.utils.utils import match_labels
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import scipy as sc
from scipy.spatial.distance import squareform
import tqdm
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.feature_selection import f_classif
from sklearn.cluster import AgglomerativeClustering

def intersect1d(array1, array2):
    # efficient using sets but only returns the elements in intersection not the positions
    set1 = set(array1)
    set2 = set(array2)
    return np.asarray(list(set1.intersection(set2)))

class GlobalStability_MP():
    def __init__(self, X, alg, n_clusters, seed = 234, **params):
        """ Initialize class with: 
        
        X : data (N, M)
        alg : base model to be used for clustering (instantiated for all parameters except n_clusters)
        n_clusters : number of clusters to use
        seed : reproducibility seed
        **params : other parameters needed
        
        """
        self.X = X
        self.model = alg
        self.n_clusters = n_clusters
        self.seed = seed 
        self.params = params

    """ 
    MPCC/IMPACC helpers
    """
    def _scale_matrix_(self, X):
        """
        Row-wise standardization (genes/features in rows, samples in columns):
        (x - mean_row) / sd_row; floor sd to avoid div-by-zero.
        """
        X = X.astype(float)
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True)
        s[s == 0] = 1e-3
        return (X - m) / s


    def _connectivity_matrix_(self, assignments, M, sample_indices):
        """
        Update N x N connectivity count matrix M with cluster assignments for a
        subset of sample indices. `assignments` is a vector of length len(sample_indices)
        with integer labels.
        """
        assignments = np.asarray(assignments).ravel()          # <- ensure 1D
        sample_indices = np.asarray(sample_indices).ravel()    # <- ensure 1D
        if assignments.size != sample_indices.size:
            raise ValueError(
                f"assignments (len {assignments.size}) and sample_indices (len {sample_indices.size}) must match."
            )
        for lab in np.unique(assignments):
            cols = sample_indices[assignments == lab]
            if cols.size == 0:
                continue
            M[np.ix_(cols, cols)] += 1
        return M

    def _consensus_confusion_(self, C):
        """
        Per-sample confusion = row mean of C * (1 - C).
        """
        return np.mean(C * (1.0 - C), axis=1)

    def _increment_cosampled_(self, M, idx):
        """
        Increment co-sampled counts for all pairs in idx (no clustering logic).
        """
        idx = np.asarray(idx).ravel()
        if idx.size:
            M[np.ix_(idx, idx)] += 1
        return M
    
    def _feature_pvals(self, submat, col_labels):
        """
        ANOVA p-values per feature (row) vs cluster labels of columns.
        Uses sklearn f_classif (returns F and p).
        """
        Xc = submat.T  # samples x features
        # guard against degenerate cases (single cluster)
        if np.unique(col_labels).size < 2 or Xc.shape[0] < 3:
            return np.full(submat.shape[0], np.nan)
        try:
            _, p = f_classif(Xc, col_labels)
        except Exception:
            p = np.full(submat.shape[0], np.nan)
        return p
        
    def _final_clusterer_(self, consensus, K, algo = "agglomerative"):
        """
        Final clustering on consensus matrix. Distance = 1 - consensus.
        """
        if K is None:
            raise ValueError("K must be provided for final clustering.")
        D = 1.0 - consensus
        if algo == "agglomerative":
            model = AgglomerativeClustering(n_clusters=K, linkage='average')
            labels = model.fit_predict(D)
        elif algo == "spectral":
            model = SpectralClustering(n_clusters=K, assign_labels='kmeans')
            labels = model.fit_predict(consensus)
        elif algo == "kmeans":
            # Embed by top-K eigenvectors of consensus (simple spectral embedding)
            w, v = np.linalg.eigh((consensus + consensus.T) / 2)
            idx = np.argsort(w)[::-1][:K]
            emb = v[:, idx]
            labels = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(emb)
        else:
            raise ValueError("Unsupported final algorithm.")
        return labels
            
    def mpcc(self, X, K, reps = 300, p_item = 0.25, p_feature = 0.10,
             base_clusterer = None, final_algo = "agglomerative", early_stop: bool = True,
             num_unchange = 5, eps = 1e-5, verbose = True):
        """
        MiniPatch Consensus Clustering with uniform sampling.
        - No dendrogram quantiles; each minipatch is clustered by `base_clusterer`.
        - Consensus matrix aggregation + optional early stopping.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array (features x samples).")
        Xs = self._scale_matrix_(X)
    
        n_feat, n_samp = Xs.shape
        if base_clusterer is None:
            base_clusterer = KMeans(n_clusters=K)
    
        # consensus bookkeeping
        Co = np.zeros((n_samp, n_samp), dtype=float)
        mCount = np.zeros_like(Co)  # times co-sampled
        ml = np.zeros_like(Co)      # times co-clustered
    
        conf_hist = []
        it = 0
        while it < reps:
            mp = adaptive_minipatch(Xs, p_item, p_feature)
            labels = base_clusterer.fit_predict(mp['submat'].T)  # labels for columns in mp.col_idx
    
            # Coassociation
            mCount = self._connectivity_matrix_(np.ones_like(labels), mCount, mp['col_idx'])
            # Co-clustering
            ml = self._connectivity_matrix_(labels, ml, mp['col_idx'])
    
            # Consensus
            with np.errstate(divide='ignore', invalid='ignore'):
                Co = np.divide(ml, mCount, out=np.zeros_like(ml, dtype=float), where=(mCount > 0))
    
            it += 1
            # early stopping on consensus confusion flattening
            if early_stop:
                conf_hist.append(np.quantile(self._consensus_confusion_(Co), 0.9))
                if len(conf_hist) > num_unchange:
                    diffs = np.abs(np.diff(conf_hist[-(num_unchange+1):]))
                    if np.max(diffs) < eps:
                        if verbose:
                            print(f"Stop at iteration {it}")
                        break
    
        labels_final = self._final_clusterer_(Co, K, algo=final_algo)
    
        return dict(
            ConsensusMatrix=Co,
            labels=labels_final,
        )
    
    def impacc(self, X, K, reps= 300, p_item = 0.25, p_feature = 0.10, adaptive_feature = True,
               qI = 0.95,          # high-uncertainty (obs) percentile
               qF = 0.95,          # high-importance (feat) percentile
               alpha_I = 0.5,      # obs weight 
               alpha_F = 0.5,      # feat weight 
               pp = 0.05,          # feature support threshold (p-value quantile)
               E = 3,               # epochs for burn-in
               base_clusterer = None, final_algo = "agglomerative", early_stop = True, 
               num_unchange = 5, eps = 1e-5, verbose = True):
        """
        Interpretable MP Adaptive Consensus Clustering:
          - Adaptive sampling over observations (always) and features (optional).
          - Feature "importance" via repeated ANOVA (f_classif) wins.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array (features x samples).")
        Xs = self._scale_matrix_(X)

        n_feat, n_samp = Xs.shape
        if base_clusterer is None:
            base_clusterer = KMeans(n_clusters=K)
    
        # burn-in: uniform sampling for coverage
        burn_iters = max(1, E)  
        Co = np.zeros((n_samp, n_samp), dtype=float)
        mCount = np.zeros_like(Co)
        ml = np.zeros_like(Co)
        
        # weights and counters
        wi = np.ones(n_samp) / n_samp             # observation weights
        wf = np.ones(n_feat) / n_feat             # feature weights (for sampling)
        feat_support = np.zeros(n_feat, dtype=float)
        feat_seen = np.zeros(n_feat, dtype=float)
        feat_score = np.zeros(n_feat, dtype=float)
    
        # schedules (pi_item, pi_feature): explore->exploit
        pi_item_sched = np.linspace(0.5, 1.0, num=reps)  # fraction of exploitation mass
        pi_feat_sched = np.linspace(0.5, 1.0, num=reps) if adaptive_feature else np.ones(reps)
        
        # --------------------
        # Burn-in (uniform)
        # --------------------
        if verbose:
            print("Burn-in stage")
        for bi in range(burn_iters):
            mp = adaptive_minipatch(Xs, p_item, p_feature)
            labels = base_clusterer.fit_predict(mp['submat'].T)
    
            labels = np.asarray(labels).ravel() 
            feat_seen[mp['row_idx']] += 1
    
            # (optional) record feature support using ANOVA p-values
            if adaptive_feature:
                pv = self._feature_pvals(mp['submat'], labels)
                if np.isfinite(pv).sum() > 0:
                    thr = np.quantile(pv[np.isfinite(pv)], pp)
                    keep = mp['row_idx'][pv <= thr]
                    feat_support[keep] += 1
                    feat_score = np.divide(feat_support, np.maximum(1, feat_seen), dtype=float)
    
            # consensus updates
            mCount = self._increment_cosampled_(mCount, mp['col_idx'])
            ml = self._connectivity_matrix_(labels, ml, mp['col_idx'])
            with np.errstate(divide='ignore', invalid='ignore'):
                Co = np.divide(ml, mCount, out=np.zeros_like(ml, dtype=float), where=(mCount > 0))
        # --------------------
        # Adaptive stage
        # --------------------
        if verbose:
            print("Adaptive stage")
    
        conf_hist = []
        it = 0
        while it < reps:
            # update obs weights from consensus confusion
            confusion = self._consensus_confusion_(Co)
            ww = confusion.copy()
            if ww.sum() > 0:
                ww = ww / ww.sum()
                wi = alpha_I * wi + (1 - alpha_I) * ww
    
            # update feature weights (EMA of feature scores)
            if adaptive_feature and feat_score.sum() > 0:
                fs = feat_score / feat_score.sum()
                wf = alpha_F * wf + (1 - alpha_F) * fs
            elif adaptive_feature:
                # no information yet; keep uniform
                wf = alpha_F * wf + (1 - alpha_F) * (np.ones_like(wf) / len(wf))
        
            # sample a minipatch with EE+Prob in both spaces
            mp = adaptive_minipatch(
                Xs, p_item, p_feature,
                weights_item=wi, weights_feature=(wf if adaptive_feature else None),
                pi_item=pi_item_sched[it],
                pi_feature=pi_feat_sched[it],
                qI=qI, qF=qF
            )
            labels = base_clusterer.fit_predict(mp['submat'].T)
    
            # feature bookkeeping + p-value support
            feat_seen[mp['row_idx']] += 1
            if adaptive_feature:
                pv = self._feature_pvals(mp['submat'], labels)
                finite = np.isfinite(pv)
                if finite.sum() > 0:
                    thr = np.quantile(pv[finite], pp)
                    keep = mp['row_idx'][pv <= thr]
                    feat_support[keep] += 1
                    feat_score = np.divide(feat_support, np.maximum(1, feat_seen), dtype=float)
        
            # consensus updates
            mCount = self._connectivity_matrix_(np.ones_like(labels), mCount, mp['col_idx'])
            ml = self._connectivity_matrix_(labels, ml, mp['col_idx'])
            with np.errstate(divide='ignore', invalid='ignore'):
                Co = np.divide(ml, mCount, out=np.zeros_like(ml, dtype=float), where=(mCount > 0))
    
            it += 1
    
            # early stopping on consensus confusion flattening
            if early_stop:
                conf_hist.append(np.quantile(self._consensus_confusion_(Co), 0.9))
                if len(conf_hist) > num_unchange:
                    diffs = np.abs(np.diff(conf_hist[-(num_unchange+1):]))
                    if np.max(diffs) < eps:
                        if verbose:
                            print(f"Stop at iteration {burn_iters + it}")
                        break
    
        labels_final = self._final_clusterer_(Co, K, algo=final_algo)
    
        return dict(
            ConsensusMatrix=Co,
            labels=labels_final,
            feature_importance=(feat_score if adaptive_feature else None),
            nIter=burn_iters + it
        )   

        
    """ 
    Model explorer helpers 
    """

    def _get_LOCO_(self, j, idx_mp_x, idx_mp_y, indices=False):
        """ Helper: LOCO-LOO-MP to get LOCO indices given j covariate to remove on minipatches 
        
        Parameters:
        __________
        j : np.int between 0 and M-1
            Index of covariate to leave out
        idx_mp_x : ndarray (B, n_B) 
            Array (or list if minipatch of varying sizes) of indices corresponding to samples from each minipatch observations
            Corresponds to (I_t) for t=1, ... B
        idx_mp_y : ndarray (B, m_B) 
            Array (or list if minipatch of varying sizes) of indices corresponding to samples from each minipatch features 
            Corresponds to (F_t) for t=1, ... B
        indices: boolean (default False)
            Set to True if want the index mapping of LOCO-MP
        Returns:
        _________        
        if indices:
            loco_mp_x: I_t minipatch indices for minipatches not containing j - observation-wise
            loco_mp_y: F_t minipatch indices for minipatches not containing j - feature-wise
            loco_mp_idx: boolean mask on 1 to B indicating whether j is sampled in minipatch
        else:
            loco_mp_x: I_t minipatch indices for minipatches not containing j - observation-wise
            loco_mp_y: F_t minipatch indices for minipatches not containing j - feature-wise
        """
        mp_without_j = [j not in idx_y for idx_y in idx_mp_y]
        if indices:
            return np.array(idx_mp_x)[mp_without_j,], np.array(idx_mp_y)[mp_without_j,], mp_without_j
        else:
            return np.array(idx_mp_x)[mp_without_j,], np.array(idx_mp_y)[mp_without_j,]            

    def _model_explorer_MP_(self, z_base, in_mp_obs, in_mp_feat, j = None):
        B = len(z_base)
        idx_obs_i, idx_feat_i, without_i = self._get_LOCO_(j, in_mp_obs, in_mp_feat, indices=True)
        ARI_score = 0
        
        if j is not None:
            for p in range(len(idx_obs_i)):
                idx_obs_b = idx_obs_i[p]
                z_b = np.array(z_base)[np.array(without_i)][p]
                for t in range(p+1, len(idx_obs_i)):
                    idx_obs_t = in_mp_obs[t]
                    z_t = np.array(z_base)[np.array(without_i)][t]
                    # check there is an intersection between indices of t and b 
                    b_inter_t, idx_b, idx_t = np.intersect1d(idx_obs_b, idx_obs_t, return_indices=True) # b_inter_t: common elements, # idx_b: indices of common elements in idx_obs_b, # idx_t: indices of common elements in idx_obs_t
                    ARI_score += adjusted_rand_score(z_b[idx_b], z_t[idx_t]) 
            B_prime = len(idx_obs_i)
            ARI_score /= B_prime*(B_prime-1)/2 
        else:
            for b in range(B):
                idx_obs_b = in_mp_obs[b]
                z_b = z_base[b]    
                for t in range(b+1, B):
                    idx_obs_t = in_mp_obs[t]
                    z_t = z_base[t]
                    # check there is an intersection
                    b_inter_t, idx_b, idx_t = np.intersect1d(idx_obs_b, idx_obs_t, return_indices=True)                    
                    ARI_score  += adjusted_rand_score(z_b[idx_b], z_t[idx_t])
            ARI_score /= B*(B-1)/2
                    
        return ARI_score
                            
    def _build_minipatches_(self, x_ratio, y_ratio, B):
        """ Helper: Build minipatches 

        Parameters:
        __________


        Returns:
        _________      
        """
        X = self.X
        K = self.n_clusters
        random_state = self.seed
        z_base = []
        in_mp_feat = []
        in_mp_obs = []

        # For each minipatch fit cluster
        for b in range(B):
            # Get all minipatches X_mp = X[np.ix_(idx_x, idx_y)], y_mp = y[np.ix_(idx_x)]
            _, _, idx_mp_x, idx_mp_y = get_minipatch(X, y_arr=None, ratio_x = x_ratio, ratio_y = y_ratio, seed=None) 
            # Get all base clusterers, randomize seed at each minipatch for model
            model = clone(self.model).set_params(random_state=random_state+b)
            z_mp = model.fit_predict(X[np.ix_(idx_mp_x, idx_mp_y)])
            z_base.append(z_mp)
            in_mp_obs.append(idx_mp_x)
            in_mp_feat.append(idx_mp_y)

        in_mp_obs = np.array(in_mp_obs)
        in_mp_feat = np.array(in_mp_feat)
        return z_base, in_mp_obs, in_mp_feat

    """ 
    Scoring functions
    """

    def _pac_score_(self, M, tau=0.1):
        tri = np.triu_indices(M.shape[0], k=1)
        vals = M[tri]
        tri_valid = np.ones_like(vals, dtype=bool)
        finite = np.isfinite(vals)
        denom = np.count_nonzero(tri_valid & finite)
        if denom==0:
            return np.nan
        numer = np.count_nonzero((vals > tau) & (vals < (1-tau)))
        return - numer/denom

    def _gini_impurity_(self, M):
        """ Averaged 1- sum_i M^2 """
        # print(1-np.mean(np.sum(M**2, axis=1)))
        return - (1-np.mean(np.sum(M**2, axis=1)))

    """
    Main routines
    """
            
    def MP_stability(self, x_ratio, y_ratio, B =100, method='model-explorer', metric='pac', plot_consensus=False):
        """ Main function to call to get minipatch stability scores

        metric include Gini impurity, pac etc.
        """
        N, d = self.X.shape[0], self.X.shape[1]
        if method=='mpcc':
            res = self.mpcc((self.X).T, K=self.n_clusters, p_item = x_ratio, p_feature = y_ratio,
            base_clusterer = self.model, final_algo = "agglomerative", early_stop = True, reps = B,
            num_unchange = 5, eps = 1e-5, verbose = True)
            
            S = res["ConsensusMatrix"]
            if plot_consensus:
                sns.heatmap(S, cmap='coolwarm')
                plt.title('Consensus matrix with MPCC')
                
            S_j = []
            for j in range(d):
                X_j = np.delete(self.X, j, axis=1)
                res = self.mpcc(X_j.T, K=self.n_clusters, p_item = x_ratio, p_feature=y_ratio, base_clusterer=self.model, reps=B)
                S_j.append(res["ConsensusMatrix"])

            if metric == 'pac':
                return np.asarray([-self._pac_score_(S_j[j]-S) for j in range(d)])
    
            elif metric == 'gini': 
                return np.asarray([-self._gini_impurity_(S_j[j]-S) for j in range(d)])
    
            else:
                print("Select a metric among 'pac', 'gini' ")  

        elif method =='impacc':
            res = self.impacc((self.X).T, K=self.n_clusters, base_clusterer= self.model, # default is KMeans
            adaptive_feature=True, reps=B, p_item=x_ratio, p_feature=y_ratio,
            final_algo = "agglomerative", early_stop = True, num_unchange = 5, eps = 1e-5, verbose = True)
            
            S = res["ConsensusMatrix"]
            if plot_consensus:
                sns.heatmap(S, cmap='coolwarm')
                plt.title('Consensus matrix with IMPACC')
                plt.show()
                sns.barplot(res['feature_importance'], orient='h')
                plt.title('Feature importance with IMPACC')
                plt.show()
                
            S_j = []
            for j in range(d):
                X_j = np.delete(self.X, j, axis=1)
                res = self.impacc(X_j.T, K=self.n_clusters, base_clusterer=self.model, reps=B, p_item=x_ratio, p_feature = y_ratio)
                S_j.append(res["ConsensusMatrix"])

            if metric == 'pac':
                return np.asarray([-self._pac_score_(S_j[j]-S) for j in range(d)])
    
            elif metric == 'gini': 
                return np.asarray([-self._gini_impurity_(S_j[j]-S) for j in range(d)])
    
            else:
                print("Select a metric among 'pac', 'gini' ")  
                
        elif method == 'model-explorer':
            print('Build minipatches and clusters')
            z_base, in_mp_obs, in_mp_feat = self._build_minipatches_(x_ratio, y_ratio, B)
            print('Compute ARI scores')            
            ARI_full = self._model_explorer_MP_(z_base, in_mp_obs, in_mp_feat)
            ARI_j = []
            for j in range(d):
                ARI_j.append(self._model_explorer_MP_(z_base, in_mp_obs, in_mp_feat, j=j))

            return np.asarray([ARI_full - ARI_j[j] for j in range(d)])
        
        else:
            print("Need to choose a method among 'model-explorer', 'mpcc', 'impacc' ")