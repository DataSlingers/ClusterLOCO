""" Cluster LOCO-MP 

Main Cluster LOCO-MP code base for LOCO feature importance with clustering generalizability via fast minipatch ensembles

Author: Claire He 
Last modification: 12/06/2026
"""
from __future__ import annotations
from typing import Any, Mapping, Optional
from joblib import Parallel, delayed
import tqdm

import numpy as np
import numpy.random as r
import pandas as pd
from random import sample

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import adjusted_rand_score

from clim.utils.utils import hungarian_align, _set_if_exists, _resolve_patch_param, _label_mapping_from_overlap, _apply_label_mapping
from .minipatch import iter_minipatches


class ClusterLOCOMP:
    """ Cluster LOCO-MP : Leave-One-Covariate-Out using clustering generalizability with Minipatch ensembles

    Cluster LOCO-MP is a feature importance score based on clustering generalizability and LOCO (Leave-One-Covariate-Out) 
    using minipath ensembles for fast computation. Minipatches are small random subsets of observations and features that
    enable embarassingly parallel model fitting. 
    
    This implementation is designed for scalability:
        - minipatch row and feature indices are stored compactly as int32 arrays;
        - fitted minipatch transfer classifiers are stored and reused
        - LOCO-LOO scores can be computed feature-wise from aggregated predictions (soft classifier) or vote counts 
        (hard classifier).

    """
    def __init__(self, K, B, base_clusterer=None, base_classifier=None, random_state=0):
        self.K = K
        self.B = B
        self.random_state = random_state
        
        # Default base clusterer/classifiers
        self.base_clusterer = clone(base_clusterer) if base_clusterer is not None else KMeans()
        self.base_classifier = clone(base_classifier) if base_classifier is not None else DecisionTreeClassifier()

        # Inner consistency of clustering and classifier parameters
        _set_if_exists(self.base_clusterer, random_state=self.random_state)
        _set_if_exists(self.base_classifier, random_state=self.random_state)

        if self.K is not None:
            _set_if_exists(self.base_clusterer, n_clusters=self.K)
            _set_if_exists(self.base_clusterer, n_components=self.K)

    def fit(self, X, y = None, alpha_N = 0.2, alpha_M = 0.2, patch_n = None, patch_m = None, 
            parallel_MP = True, parallel: Optional[Mapping[str, Any]] = None, par = None, 
            pprint: bool = True, reference: str = 'full', min_known: int = 10, standardize = False, 
           ):
        X = np.asarray(X)
        B = self.B
        N, M = X.shape
        self.N_, self.M_ = N, M
        rng = np.random.RandomState(self.random_state)
        if par is None:
            par = {"n_jobs": -1, "backend": "loky", "prefer": "processes", "verbose": 0,}

        self.parallel_MP = parallel_MP
        # allow either fractional patch sizes (alpha_N, alpha_M) or explicit patch sizes (patch_n, patch_m). 
        # Explicit sizes take precedence for patch size control
        n = _resolve_patch_param(alpha_N, patch_n, N, "alpha_N", "patch_n")
        m = _resolve_patch_param(alpha_M, patch_m, M, "alpha_M", "patch_m")
    
        # Store both raw inputs and resolved sizes
        self.alpha_N, self.alpha_M = alpha_N, alpha_M
        self.patch_n, self.patch_m = patch_n, patch_m
        self.patch_n_, self.patch_m_ = n, m

        # Prepare data 
        self.standardize = standardize
        self.scaler = StandardScaler()
        Xs = np.asarray(self.scaler.fit_transform(X), dtype=np.float32) if standardize else np.asarray(X, dtype=np.float32)
        
        # Prepare parallelization settings
        if parallel: 
            par.update(parallel)
            
        if pprint:
            print("---------- Generate minipatches ------------")
        # Compact patch storage via dense int32 arrays instead of list of tuples
        patches_gen = iter_minipatches(N, M, self.B, n, m, rng=rng, sort_indices=False)
        self.patch_rows_ = np.empty((self.B, n), dtype=np.int32)
        self.patch_cols_ = np.empty((self.B, m), dtype=np.int32)
        for b, I_t, F_t in patches_gen:
            self.patch_rows_[b] = I_t
            self.patch_cols_[b] = F_t

        self._build_feature_to_patches(M) # prepare look up table 
        
        if pprint:
            print("---------- Minipatch clustering ------------")
        if parallel_MP:
            mp_labels = Parallel(n_jobs=par['n_jobs'], backend=par['backend'], verbose=par['verbose'], prefer=par['prefer'],)(
                delayed(self._mp_cluster)(Xs, self.patch_rows_[b], self.patch_cols_[b],) for b in range(self.B))
            self.mp_labels_raw_ = np.vstack(mp_labels).astype(np.int32)
        else:
            self.mp_labels_raw_ = np.vstack([self._mp_cluster(Xs, self.patch_rows_[b], self.patch_cols_[b],) for b in range(self.B)]).astype(np.int32)
            
        if self.K is None:
            self.K = int(np.max(self.mp_labels_raw_)) + 1
        else:
            self.K = int(self.K)
                
        if pprint:
            print("---------- Minipatch reference ------------")
        
        reference = reference.lower()
        ### Main pipeline 
        if reference == "full":
            # Construct reference 
            z_ref = self._cluster_fit_predict(self.base_clusterer, Xs)
            self.z_ref = np.asarray(z_ref, dtype=np.int32)
        
            if np.any(self.z_ref < 0):
                raise ValueError(
                    "Cluster labels must be nonnegative integers. "
                    "Use sklearn_wrappers mapping DBSCAN noise labels to a valid cluster.")
            self.K = int(np.max(self.z_ref)) + 1 if self.K is None else int(self.K)
            self.mp_labels_ = np.empty_like(self.mp_labels_raw_)
            for b in range(self.B):
                self.mp_labels_[b] = hungarian_align(self.z_ref[self.patch_rows_[b]], self.mp_labels_raw_[b]).astype(np.int32, copy=False)

        ### Alternative alignment schemes
        elif reference == "online":
            self.z_ref, self.mp_labels_ = self._online_consensus_reference(min_known=min_known)

        elif reference == "consensus":
            self.z_ref, self.mp_labels_ = self._full_coassociation_reference()
        
        else:
            raise ValueError("reference must be one of {'full', 'online', 'consensus'}.")
        
        if pprint:
            print("---------- Minipatch training ------------")

        if parallel_MP:
            self.models_ = Parallel(n_jobs=par['n_jobs'], backend=par['backend'], verbose=par['verbose'], prefer=par['prefer'],)(
                    delayed(self._mp_transfer)(Xs, b) for b in range(self.B))
        else: # sequential MP processing
            self.models_ = []
            for b in range(self.B):
                self.models_.append(self._mp_transfer(Xs, b))

        self.X_fit_ = Xs
        return self

    ## Clustering generalizability training helpers 
    def _mp_cluster(self, Xs, I_t, F_t):
        """ 
        Cluster one minipatch and return patch unaligned label
        Returns
        -------
        z_b : ndarray, shape (|I_t|, ) 
            cluster labels for minipatch b
        """
        X_mp = Xs[np.ix_(I_t, F_t)]
        z_b = np.asarray(self._cluster_fit_predict(self.base_clusterer, X_mp), dtype=np.int32)
        return z_b

    def _mp_transfer(self, X, b):
        """
        Train minipatch classifier with aligned minipatch labels
        Requires
        --------
        X: full data 
        b: minipatch index
        aligned: bool, if labels have been aligned 
        Returns
        -------
        f_mp : a fitted sklearn minipatch specific classifier 
        """
        N, K = self.N_, self.K
        I_b, F_b = self.patch_rows_[b], self.patch_cols_[b]
        z_mp= self.mp_labels_[b]
        X_mp = X[np.ix_(I_b, F_b)]        
        f_mp = clone(self.base_classifier)
        f_mp.fit(X_mp, z_mp)
        return f_mp
        
    def _cluster_fit_predict(self, clusterer, X):
        """
        Return cluster labels for X using the broadest sklearn-compatible interface.
        Supports clusterers with:
        - fit_predict(X)
        - fit(X).labels_
        - fit(X).predict(X)
        """
        C = clone(clusterer)
        if hasattr(C, "fit_predict"):
            z = C.fit_predict(X)
        else:
            C.fit(X)
            if hasattr(C, "labels_"):
                z = C.labels_
            elif hasattr(C, "predict"):
                z = C.predict(X)
            else:
                raise TypeError(
                    f"{C.__class__.__name__} must implement either "
                    "fit_predict(X), fit(X).labels_, or fit(X).predict(X).")
        return np.asarray(z, dtype=np.int32)

    ## LOCO-LOO/LOO aggregation helpers 
    def _mp_predict_omp(self, Xs, b, proba):
        """
        Predict OMP (out-of-minipatch) probabilities/labels for stored minipatch model b.
        """
        N, K = self.N_, self.K
        f_mp = self.models_[b] # stored minipatch model b

        # Out-of-MiniPatch data
        omp_mask = np.ones(N, dtype=bool)
        omp_mask[self.patch_rows_[b]] = False
        omp_rows = np.flatnonzero(omp_mask).astype(np.int32, copy=False)
        X_omp = Xs[np.ix_(omp_rows, self.patch_cols_[b])]
        if proba:
            p_b = f_mp.predict_proba(X_omp).astype(np.float32, copy=False)
            p_omp = self._align_mp_proba_to_classes(p_b, f_mp.classes_)
            return omp_rows, p_omp
        else:
            z_omp = f_mp.predict(X_omp).astype(np.int32, copy=False)
            return omp_rows, z_omp

    def _mp_loo(self, Xs, proba=True, patches=None):
        """
        Minipatch LOO aggregator 
        Requires
        --------
        Xs: full data
        proba: if using soft classifier when True
        patches: if LOO, set to None, otw skips LOCO patches
        
        Returns
        -------
        loo_proba/loo_sum/loo_count if proba
        loo_labels/vote_count/loo_count otw
        """
        N, K = self.N_, self.K
        if patches is None:
            patches = np.arange(self.B, dtype=np.int32)
        else:
            patches = np.asarray(patches, dtype=np.int32)
        
        out = [self._mp_predict_omp(Xs, b, proba) for b in patches]

        if proba:
            loo_sum, loo_count = np.zeros((N, K), dtype=np.float32), np.zeros(N, dtype=np.int32)
            for rows, p_mp in out:
                loo_sum[rows]+=p_mp
                loo_count[rows]+=1      
            loo_proba = np.full((N, K), np.nan, dtype=np.float32)
            ok = loo_count > 0
            loo_proba[ok] = loo_sum[ok] / loo_count[ok, None]
            return loo_proba, loo_sum, loo_count

        else: # use max voting on hard labels 
            vote_count = np.zeros((N, K), dtype=np.int32)
            loo_count = np.zeros(N, dtype=np.int32)
            for rows, z_mp in out:
                valid = (z_mp >= 0) & (z_mp < K)
                rows_valid = rows[valid]
                z_valid = z_mp[valid]
                np.add.at(vote_count, (rows_valid, z_valid), 1)
                loo_count[rows_valid] += 1
            loo_labels = np.full(N, -1, dtype=np.int32)
            ok = loo_count > 0
            loo_labels[ok] = np.argmax(vote_count[ok], axis=1).astype(np.int32)
            return loo_labels, vote_count, loo_count     

    def _mp_loco_loo(self, Xs, j, loo_sum, loo_count, proba=True):
        """
        Compute LOCO-LOO aggregate for feature j.
    
        LOCO-LOO(-j) uses patches satisfying:
            i not in I_b and j not in F_b.
    
        Since LOO uses all patches with i not in I_b:
            LOCO(-j) = LOO - contribution from patches that included j.

        Requires
        --------
        Xs: data
        j: feature j to remove in LOCO-LOO
        proba: if true, work with classifier probability, if not use hard labels
        loo_sum: if proba is true, loo_sum, otw vote_count 
        loo_count: loo counts
        """    
        N, K = self.N_, self.K
        patches_with_j = self.feature_to_patches_[j]
        if proba:
            _, included_sum, included_count = self._mp_loo(Xs, proba, patches=patches_with_j)
            # subtractive trick LOO = LOCO_LOO(j) + LOO_with_j
            loco_sum = loo_sum - included_sum
            loco_count = loo_count - included_count
            loco_proba = np.full((N, K), np.nan, dtype=np.float32)
            ok = loco_count > 0
            loco_proba[ok] = loco_sum[ok] / loco_count[ok, None]
            return loco_proba, loco_sum, loco_count
        else:
            _, included_vote_count, included_count = self._mp_loo(Xs, proba=False, patches=patches_with_j)
            loco_vote_count = loo_sum - included_vote_count
            loco_count = loo_count - included_count
            loco_labels = np.full(N, -1, dtype=np.int32)
            ok = loco_count > 0 
            loco_labels[ok] = np.argmax(loco_vote_count[ok], axis=1).astype(np.int32)
            return loco_labels, loco_vote_count, loco_count
            

    ## Alignment helpers 
    def _full_coassociation_reference(self):
        """
        Build a reference labeling from the full co-association matrix.
        For observations i and j,
            S[i, j] = (# patches where i and j are together and co-clustered) / (# patches where i and j are together)
        Then cluster distance D = 1 - S.
    
        Warning
        -------
        This uses O(N^2) memory and O(B n^2) time. It is only appropriate
        for moderate N.
    
        Requires
        --------
        self.patch_rows_ : ndarray, shape (B, n)
        self.mp_labels_raw : ndarray, shape (B, n)
        self.K : int
        self.N : int
    
        Returns
        -------
        z_ref : ndarray, shape (N,)
            Consensus reference labels.
        patch_labels_aligned : ndarray, shape (B, n)
            Patch labels aligned to z_ref.
        """
        B = self.B
        n = self.patch_rows_.shape[1]
        N = int(self.N)
        K = int(self.K)
    
        count_dtype = np.uint16 if B <= np.iinfo(np.uint16).max else np.uint32
        same_count = np.zeros((N, N), dtype=count_dtype)
        co_count = np.zeros((N, N), dtype=count_dtype)
    
        for b in range(B):
            I_b = self.patch_rows_[b]
            z_b = self.patch_labels_raw_[b]
    
            # All pairs in the same minipatch co-occur.
            co_count[np.ix_(I_b, I_b)] += 1
    
            # Pairs assigned to the same local cluster co-cluster.
            for k in range(K):
                members = I_b[z_b == k]
                if members.size > 0:
                    same_count[np.ix_(members, members)] += 1
    
        S = np.zeros((N, N), dtype=np.float32)
        ok = co_count > 0
        S[ok] = same_count[ok] / co_count[ok]
    
        # For observations with no co-occurrence information with others,
        # at least set the diagonal to 1.
        np.fill_diagonal(S, 1.0)
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)

        try:
            clusterer = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average",)
        except TypeError:
            clusterer = AgglomerativeClustering(n_clusters=K, affinity="precomputed", linkage="average")
    
        z_ref = np.asarray(clusterer.fit_predict(D), dtype=np.int32)
        patch_labels_aligned = np.empty_like(self.mp_labels_raw_)
        for b in range(B):
            patch_labels_aligned[b] = hungarian_align(z_ref[self.patch_rows_[b]], self.mp_labels_raw_[b])
    
        return z_ref, patch_labels_aligned

    def _online_consensus_reference(self, min_known=10, order=None):
        """
        Build a reference labeling from minipatch overlaps using online consensus.
    
        This does not require a full-data clustering. It processes minipatches
        sequentially. For each patch, labels are aligned to the current global
        consensus using observations that have already appeared in previous patches.

        This is a faster alignment scheme with time complexity of O(B(n + K^3)) and memory O(NK+BK)
    
        Requires
        --------
        self.patch_rows_ : ndarray, shape (B, n)
            minipatch row (observation) index
        self.mp_labels_raw : ndarray, shape (B, n)
            raw labels obtained after clustering minipatches 
        self.K : int
        self.N : int
    
        Parameters
        ----------
        min_known : int, default=10
            Minimum number of already-seen observations in a patch required
            to estimate a Hungarian alignment. If fewer are available, the
            identity mapping is used.
        order : array-like or None
            Optional order in which to process patches. If None, use 0,...,B-1.
    
        Returns
        -------
        z_ref : ndarray, shape (N,)
            Online consensus reference labels.
        mp_labels : ndarray, shape (B, n)
            Aligned minipatch labels.
        """
        B = self.B
        n = self.patch_rows_.shape[1]
        N = int(self.N)
        K = int(self.K)
        if order is None:
            order = np.arange(B, dtype=np.int32)
        else:
            order = np.asarray(order, dtype=np.int32)
    
        global_votes = np.zeros((N, K), dtype=np.int32)
        seen_counts = np.zeros(N, dtype=np.int32)
        mp_labels = np.full((B, n), -1, dtype=np.int32)
    
        for b in order:
            b = int(b)
            I_b, z_b = self.patch_rows_[b], self.mp_labels_raw_[b]
            known = seen_counts[I_b] > 0
            n_known = int(np.sum(known))
    
            if n_known >= min_known:
                z_ref_known = global_votes[I_b[known]].argmax(axis=1) # Current consensus labels on observations already seen
                z_local_known = z_b[known] # Local labels for the same observations
                mapping = _label_mapping_from_overlap(K, z_ref_known, z_local_known)
            else:
                mapping = np.arange(K, dtype=np.int32)
    
            z_b_aligned = _apply_label_mapping(z_b, mapping)
            mp_labels[b] = z_b_aligned
    
            valid = z_b_aligned >= 0
            np.add.at(global_votes, (I_b[valid], z_b_aligned[valid]), 1)
            seen_counts[I_b[valid]] += 1
    
        z_ref = np.full(N, -1, dtype=np.int32)
        seen = seen_counts > 0
        z_ref[seen] = global_votes[seen].argmax(axis=1)
        return z_ref, mp_labels

    ### Additional helpers 
    def _build_feature_to_patches(self, M=None):
        """
        Build reverse lookup:
            feature_to_patches_[j] = array of patch indices b such that j in patch_cols_[b]

        This is used for LOCO-LOO:
            LOCO(-j) = LOO - contributions from patches that included j
        """
        if M is None:
            M = int(self.M_)
        feature_to_patches = [[] for _ in range(M)]
    
        for b in range(int(self.B)):
            for j in self.patch_cols_[b]:
                feature_to_patches[int(j)].append(b)
        self.feature_to_patches_ = [np.asarray(patches, dtype=np.int32) for patches in feature_to_patches]
        return self.feature_to_patches_
    

    def _align_mp_proba_to_classes(self, p_pred, classes_):
        """
        Map minipatch classifier probabilities (K_seen values) into global 0...K-1 columns.
        Returns (N, K) float32.
        """
        N = p_pred.shape[0]
        Pglob = np.zeros((N, self.K), dtype=np.float32)
        cls = np.asarray(classes_, dtype=int)

        for j, c in enumerate(cls):
            if 0 <= c < self.K:
                Pglob[:, c] = p_pred[:, j]
        return Pglob


    ## Computer Cluster LOCO-MP
    def score(self, error_metric=None, z=None, X=None, agg="mean", features=None, proba_error=True, parallel_features: bool = False, par=None):
        """
        Compute Cluster LOCO-MP feature importance scores.
    
        Supported modes
        ---------------
        1. error_metric is None:
            Uses adjusted Rand index with hard-label predictions.
    
            delta_j = ARI(LOO) - ARI(LOCO without feature j)
    
        2. error_metric is not None and proba_error=True:
            Uses probability predictions.
    
            error_metric(z, P) must return pointwise errors of shape (N,).
    
            delta_j = error(LOCO without feature j) - error(LOO)
    
        3. error_metric is not None and proba_error=False:
            Uses hard-label predictions.
    
            error_metric(z, z_hat) may return either:
                - scalar error, e.g. hamming_distance
                - pointwise errors of shape (N,)
    
            delta_j = error(LOCO without feature j) - error(LOO)
    
        Notes
        -----
        Patch-level parallelism is deliberately disabled inside score().
        If parallel_features=True, parallelization is over features instead.
        """
        import numpy as np
        import pandas as pd
        from joblib import Parallel, delayed
        from sklearn.metrics import adjusted_rand_score
        from sklearn.exceptions import NotFittedError
    
        if par is None:
            par = {
                "n_jobs": -1,
                "backend": "loky",
                "prefer": "processes",
                "verbose": 0,
            }
    
        if agg not in {"mean", "none", "by_clusters"}:
            raise ValueError("agg must be one of {'mean', 'none', 'by_clusters'}.")
    
        # ------------------------------------------------------------
        # Prepare data.
        # ------------------------------------------------------------
        if X is None:
            if not hasattr(self, "X_fit_"):
                raise NotFittedError("Run fit() before calling score().")
            Xs = self.X_fit_
        else:
            X = np.asarray(X)
            Xs = np.asarray(self.scaler.transform(X), dtype=np.float32) if self.standardize else np.asarray(X, dtype=np.float32)
    
        if z is None:
            z = self.z_ref
        z = np.asarray(z)
    
        N, M = Xs.shape
    
        if z.shape[0] != N:
            raise ValueError(
                f"z has length {z.shape[0]}, but X has {N} rows. "
                "For LOO/LOCO-LOO scoring, z must match the scored data."
            )
    
        if features is None:
            feats = np.arange(M, dtype=np.int32)
        else:
            feats = np.asarray(features, dtype=np.int32)
    
        F = len(feats)
    
        # ------------------------------------------------------------
        # Helper: cluster aggregation.
        # ------------------------------------------------------------
        def _cluster_aggregate(delta_all, labels):
            labels = np.asarray(labels)
            cluster_ids = pd.unique(labels)
            Kc = len(cluster_ids)
    
            delta_by_cluster = np.full((F, Kc), np.nan, dtype=np.float32)
    
            for c_pos, cl in enumerate(cluster_ids):
                mask = labels == cl
                if np.any(mask):
                    delta_by_cluster[:, c_pos] = np.nanmean(delta_all[:, mask], axis=1)
    
            delta_mean = np.nanmean(delta_by_cluster, axis=1)
            n_eff = np.sum(~np.isnan(delta_by_cluster), axis=1)
    
            delta_se = np.full(F, np.nan, dtype=np.float32)
            ok = n_eff > 1
            delta_se[ok] = (
                np.nanstd(delta_by_cluster[ok], axis=1, ddof=1)
                / np.sqrt(n_eff[ok])
            )
    
            return cluster_ids, delta_by_cluster, delta_mean, delta_se
    
        # --------------------------------------
        # Case 1: ARI mode.
        # --------------------------------------
        if error_metric is None:
            metric_name = "adjusted_rand_score"
    
            # Baseline LOO once. Patch-level parallelism disabled.
            loo_pred, loo_vote_count, loo_count_base = self._mp_loo(Xs, proba=False, patches=None)
    
            loo_score = float(adjusted_rand_score(z, loo_pred))
    
            def _score_one_feature_ari(f_pos, j):
                loco_pred, loco_vote_count, loco_count = self._mp_loco_loo(Xs, j=int(j), loo_sum=loo_vote_count, loo_count=loo_count_base, proba=False)
                loco_score = float(adjusted_rand_score(z, loco_pred))
                delta_j = loo_score - loco_score
                return f_pos, delta_j, loco_score
    
            if parallel_features:
                results = Parallel(
                    n_jobs=par["n_jobs"],
                    backend=par["backend"],
                    prefer=par["prefer"],
                    verbose=par["verbose"],
                )(
                    delayed(_score_one_feature_ari)(f_pos, int(j))
                    for f_pos, j in enumerate(feats)
                )
            else:
                results = [_score_one_feature_ari(f_pos, int(j)) for f_pos, j in tqdm.tqdm(enumerate(feats))]
    
            delta = np.full(F, np.nan, dtype=np.float32)
            loco_value = np.full(F, np.nan, dtype=np.float32)
    
            for f_pos, delta_j, loco_score in results:
                delta[f_pos] = delta_j
                loco_value[f_pos] = loco_score
    
            return {
                "metric": metric_name,
                "features": feats,
                "delta": delta,
                "delta_se": None,
                "loo_value": loo_score,
                "loco_value": loco_value,
                "loo_pred": loo_pred,
                "loo_count": loo_count_base,
                "proba": False,
            }
    
        # --------------------------------------
        # Case 2/3: External error metric.
        # --------------------------------------
        metric_name = getattr(error_metric, "__name__", "error_metric")
        use_proba = bool(proba_error)
    
        # Baseline LOO once. Patch-level parallelism disabled.
        pred_loo, loo_sum, loo_count_base = self._mp_loo(Xs, proba=use_proba, patches=None)
        loo_vals = np.asarray(error_metric(z, pred_loo))
    
        # ------------------------------------------------------------
        # Case 2a/3a: scalar error, e.g. hamming_distance.
        # ------------------------------------------------------------
        if loo_vals.ndim == 0:
            loo_scalar = float(loo_vals)
    
            def _score_one_feature_scalar(f_pos, j):
                pred_loco, loco_sum, loco_count = self._mp_loco_loo(Xs, j=int(j), loo_sum=loo_sum, loo_count=loo_count_base, proba=use_proba)
                loco_scalar = float(np.asarray(error_metric(z, pred_loco)))
                delta_j = loco_scalar - loo_scalar
                return f_pos, delta_j, loco_scalar
    
            if parallel_features:
                results = Parallel(
                    n_jobs=par["n_jobs"],
                    backend=par["backend"],
                    prefer=par["prefer"],
                    verbose=par["verbose"],
                )(
                    delayed(_score_one_feature_scalar)(f_pos, int(j))
                    for f_pos, j in enumerate(feats)
                )
            else:
                results = [_score_one_feature_scalar(f_pos, int(j)) for f_pos, j in tqdm.tqdm(enumerate(feats))]
    
            delta = np.full(F, np.nan, dtype=np.float32)
            loco_value = np.full(F, np.nan, dtype=np.float32)
    
            for f_pos, delta_j, loco_scalar in results:
                delta[f_pos] = delta_j
                loco_value[f_pos] = loco_scalar
    
            return {
                "metric": metric_name,
                "features": feats,
                "delta": delta,
                "delta_se": None,
                "loo_value": loo_scalar,
                "loco_value": loco_value,
                "loo_pred": pred_loo,
                "loo_count": loo_count_base,
                "proba": use_proba,
            }
    
        # ------------------------------------------------------------
        # Case 2b/3b: pointwise error.
        # ------------------------------------------------------------
        if loo_vals.shape != (N,):
            raise ValueError(
                f"error_metric must return either a scalar or pointwise errors "
                f"of shape ({N},). Got shape {loo_vals.shape}."
            )
    
        ok_loo = loo_count_base > 0
    
        def _score_one_feature_pointwise(f_pos, j):
            pred_loco, loco_sum, loco_count = self._mp_loco_loo(Xs, j=int(j), loo_sum=loo_sum, loo_count=loo_count_base, proba=use_proba,)
            loco_vals = np.asarray(error_metric(z, pred_loco))
    
            if loco_vals.shape != (N,):
                raise ValueError(
                    f"error_metric must return pointwise errors of shape ({N},). "
                    f"Got shape {loco_vals.shape}."
                )
    
            valid = (ok_loo & (loco_count > 0) & ~np.isnan(loo_vals) & ~np.isnan(loco_vals))
    
            diff = np.full(N, np.nan, dtype=np.float32)
            diff[valid] = loco_vals[valid] - loo_vals[valid]
    
            if np.any(valid):
                diff_valid = diff[valid]
                delta_mean_j = float(np.nanmean(diff_valid))
                loco_value_j = float(np.nanmean(loco_vals[valid]))
                n_eff_j = int(np.sum(valid))
    
                if n_eff_j > 1:
                    delta_se_j = float(
                        np.nanstd(diff_valid, ddof=1) / np.sqrt(n_eff_j)
                    )
                else:
                    delta_se_j = np.nan
            else:
                delta_mean_j = np.nan
                delta_se_j = np.nan
                loco_value_j = np.nan
                n_eff_j = 0
    
            return f_pos, diff, delta_mean_j, delta_se_j, loco_value_j, n_eff_j
    
        if parallel_features:
            results = Parallel(
                n_jobs=par["n_jobs"],
                backend=par["backend"],
                prefer=par["prefer"],
                verbose=par["verbose"],
            )(
                delayed(_score_one_feature_pointwise)(f_pos, int(j))
                for f_pos, j in enumerate(feats)
            )
        else:
            results = [_score_one_feature_pointwise(f_pos, int(j)) for f_pos, j in tqdm.tqdm(enumerate(feats))]
    
        delta_mean = np.full(F, np.nan, dtype=np.float32)
        delta_se = np.full(F, np.nan, dtype=np.float32)
        loco_value = np.full(F, np.nan, dtype=np.float32)
        n_eff = np.zeros(F, dtype=np.int32)
    
        if agg in {"none", "by_clusters"}:
            delta_all = np.full((F, N), np.nan, dtype=np.float32)
        else:
            delta_all = None
    
        for f_pos, diff, delta_mean_j, delta_se_j, loco_value_j, n_eff_j in results:
            delta_mean[f_pos] = delta_mean_j
            delta_se[f_pos] = delta_se_j
            loco_value[f_pos] = loco_value_j
            n_eff[f_pos] = n_eff_j
    
            if agg in {"none", "by_clusters"}:
                delta_all[f_pos] = diff
    
        if agg == "none":
            return {
                "metric": metric_name,
                "features": feats,
                "delta": delta_all,
                "delta_se": None,
                "loo_pred": pred_loo,
                "loo_count": loo_count_base,
                "proba": use_proba,
            }
    
        if agg == "mean":
            return {
                "metric": metric_name,
                "features": feats,
                "delta": delta_mean,
                "delta_se": delta_se,
                "loco_value": loco_value,
                "n_eff": n_eff,
                "loo_pred": pred_loo,
                "loo_count": loo_count_base,
                "proba": use_proba,
            }
    
        if agg == "by_clusters":
            cluster_ids, delta_by_cluster, delta_cluster_mean, delta_cluster_se = (
                _cluster_aggregate(delta_all, z)
            )
    
            return {
                "metric": metric_name,
                "features": feats,
                "delta": delta_cluster_mean,
                "delta_se": delta_cluster_se,
                "delta_by_cluster": delta_by_cluster,
                "clusters": cluster_ids,
                "loo_pred": pred_loo,
                "loo_count": loo_count_base,
                "proba": use_proba,
            }


    def _prepare_score_data(self, X=None):
        if X is None:
            if not hasattr(self, "X_fit_"):
                raise NotFittedError("Run fit() before prediction or scoring.")
            return self.X_fit_
    
        X = np.asarray(X)
        return (
            np.asarray(self.scaler.transform(X), dtype=np.float32)
            if self.standardize
            else np.asarray(X, dtype=np.float32)
        )
    
    
    def predict(self, X=None, par=None):
        """
        Predict labels by LOO max-vote aggregation over stored minipatch models.
        """
        if par is None:
            par = {"n_jobs": -1, "backend": "loky", "prefer": "processes", "verbose": 0}
    
        Xs = self._prepare_score_data(X)
        z_hat, vote_count, pred_count = self._mp_loo(
            Xs,
            par=par,
            proba=False,
            patches=None,
        )
        return z_hat
    
    
    def predict_proba(self, X=None, par=None):
        """
        Predict probabilities by LOO probability aggregation over stored minipatch models.
        """
        if par is None:
            par = {"n_jobs": -1, "backend": "loky", "prefer": "processes", "verbose": 0}
    
        Xs = self._prepare_score_data(X)
        P_hat, pred_sum, pred_count = self._mp_loo(
            Xs,
            par=par,
            proba=True,
            patches=None,
        )
        return P_hat