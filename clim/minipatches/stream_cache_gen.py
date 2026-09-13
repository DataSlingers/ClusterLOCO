"""Streaming minipatch fitting and prediction-cached LOCO scoring.

Example
-------
>>> from clim.minipatches.stream_cache_gen import ClusterLOCOMPStream
>>> from clim.utils import hinge_error
>>> model = ClusterLOCOMPStream(K=7, B=5000)
>>> model.fit_stream(X, alpha_N=0.2, alpha_M=0.2, n_jobs=4,
...                  patch_batch_size=16, standardize=True)  # doctest: +SKIP
>>> scores = model.score(hinge_error, max_cache_bytes=256 * 1024**2,
...                      cache_dir="/tmp")  # doctest: +SKIP

The inherited ``fit`` is also supported. ``fit_stream`` uses a full-data
reference and generates, clusters, aligns and trains patches in bounded batches.
It still retains fitted models and patch indices, which are needed for reuse.

Scoring predicts once per patch per call. The cache contains out-of-patch rows
and probabilities (or hard labels), with no feature dimension. ``cache='auto'``
uses RAM up to ``max_cache_bytes`` and otherwise uses an automatically closed
temporary binary file. Disk records are read one patch at a time, not memory
mapped. The cache budget excludes fitted models, data, working arrays and output.
Aggregation needs O(N*K) working arrays plus O(F) output for mean scoring or
O(F*K) output for cluster aggregation. Predicting one patch also requires its
OMP feature matrix, O((N-n)*m). Only ``agg='none'`` retains an O(F*N) result.
Scoring is sequential to bound memory and avoid competing reads of the cache.

LOO scoring and prediction require the training observations in their original
order. An explicit X may alter values but must keep the fitted shape and order.
"""

from itertools import islice
from numbers import Integral
from tempfile import TemporaryFile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .generalizability import ClusterLOCOMP
from .minipatch import iter_minipatches
from clim.utils.utils import _resolve_patch_param, hungarian_align

__all__ = ["ClusterLOCOMPStream"]


def _fit_patch(clusterer, classifier, X, z_ref, rows, cols):
    """A small worker payload, without serializing the fitted ensemble."""
    patch = X[np.ix_(rows, cols)]
    clusterer = clone(clusterer)
    if hasattr(clusterer, "fit_predict"):
        raw = clusterer.fit_predict(patch)
    else:
        clusterer.fit(patch)
        raw = (clusterer.labels_ if hasattr(clusterer, "labels_")
               else clusterer.predict(patch))
    raw = np.asarray(raw, dtype=np.int32)
    aligned = hungarian_align(z_ref[rows], raw).astype(np.int32)
    model = clone(classifier).fit(patch, aligned)
    return raw, aligned, model


class _PredictionCache:
    """Fixed-size records, owned exclusively by one score call."""

    def __init__(self, B, n_omp, K, proba, mode, budget, directory):
        if mode not in {"auto", "memory", "disk"}:
            raise ValueError("cache must be 'auto', 'memory', or 'disk'")
        if not isinstance(budget, Integral) or budget < 0:
            raise ValueError("max_cache_bytes must be a nonnegative integer")
        self.dtype = np.dtype([
            ("rows", np.int32, (n_omp,)),
            ("pred", np.float32 if proba else np.int32,
             (n_omp, K) if proba else (n_omp,)),
        ])
        self.nbytes = int(B) * self.dtype.itemsize
        self.mode = ("memory" if self.nbytes <= budget else "disk") if mode == "auto" else mode
        if self.mode == "memory" and self.nbytes > budget:
            raise ValueError("Prediction cache exceeds max_cache_bytes; use cache='auto' or 'disk'")
        self.B, self.directory = B, directory
        self.records = self.file = None

    def __enter__(self):
        if self.mode == "memory":
            self.records = np.empty(self.B, dtype=self.dtype)
        else:
            self.file = TemporaryFile(dir=self.directory)
        return self

    def __exit__(self, *exc):
        self.records = None
        if self.file is not None:
            self.file.close()

    def put(self, b, rows, pred):
        record = np.empty(1, dtype=self.dtype)
        record["rows"][0], record["pred"][0] = rows, pred
        if self.records is not None:
            self.records[b] = record[0]
        elif self.dtype.itemsize:
            self.file.seek(b * self.dtype.itemsize)
            record.tofile(self.file)

    def get(self, b):
        if self.records is not None:
            record = self.records[b]
        elif self.dtype.itemsize:
            self.file.seek(int(b) * self.dtype.itemsize)
            record = np.fromfile(self.file, dtype=self.dtype, count=1)[0]
        else:  # Patches contain every observation: no OMP predictions.
            record = np.empty(1, dtype=self.dtype)[0]
        return record["rows"], record["pred"]


def _add(sums, counts, rows, pred, proba):
    if proba:
        sums[rows] += pred
    else:
        valid = (pred >= 0) & (pred < sums.shape[1])
        rows, pred = rows[valid], pred[valid]
        np.add.at(sums, (rows, pred), 1)
    counts[rows] += 1


def _prediction(sums, counts, proba):
    ok = counts > 0
    if proba:
        pred = np.full(sums.shape, np.nan, dtype=np.float32)
        pred[ok] = sums[ok] / counts[ok, None]
    else:
        pred = np.full(counts.shape, -1, dtype=np.int32)
        pred[ok] = np.argmax(sums[ok], axis=1)
    return pred


class ClusterLOCOMPStream(ClusterLOCOMP):
    """ClusterLOCOMP with bounded patch streams and a per-call prediction cache.

    Constructor and inherited fit parameters match ClusterLOCOMP. Use
    ``fit_stream`` for bounded fitting; ``iter_patch_predictions`` exposes a
    lazy prediction stream. Random-state-controlled estimators give equivalent
    full-reference fits, although reference fitting occurs before patch fitting.
    """

    def fit_stream(self, X, y=None, *, alpha_N=0.2, alpha_M=0.2,
                   patch_n=None, patch_m=None, standardize=False,
                   n_jobs=1, patch_batch_size=16):
        """Fit against a full reference, retaining at most one batch of tasks.

        ``n_jobs`` controls loky workers; native threads are limited to one.
        ``patch_batch_size`` bounds in-flight patches, independently of B.
        Patch indices, raw/aligned labels and fitted classifiers remain stored.
        """
        X = np.asarray(X)
        if X.ndim != 2 or min(X.shape) == 0:
            raise ValueError("X must be a nonempty two-dimensional array")
        if not isinstance(self.B, Integral) or self.B < 1:
            raise ValueError("B must be a positive integer")
        if not isinstance(patch_batch_size, Integral) or patch_batch_size < 1:
            raise ValueError("patch_batch_size must be a positive integer")
        if not isinstance(n_jobs, Integral) or n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer")
        # A failed refit must not leave the previous data marked as fitted.
        self.__dict__.pop("X_fit_", None)
        N, M = X.shape
        n = _resolve_patch_param(alpha_N, patch_n, N, "alpha_N", "patch_n")
        m = _resolve_patch_param(alpha_M, patch_m, M, "alpha_M", "patch_m")
        self.N_, self.M_ = N, M
        self.alpha_N, self.alpha_M = alpha_N, alpha_M
        self.patch_n, self.patch_m = patch_n, patch_m
        self.patch_n_, self.patch_m_ = n, m
        self.standardize, self.scaler = standardize, StandardScaler()
        Xs = np.asarray(self.scaler.fit_transform(X) if standardize else X, dtype=np.float32)
        with threadpool_limits(limits=1):
            self.z_ref = self._cluster_fit_predict(self.base_clusterer, Xs)
        if np.any(self.z_ref < 0):
            raise ValueError("Cluster labels must be nonnegative integers")
        self.K = int(self.z_ref.max()) + 1 if self.K is None else int(self.K)
        self.patch_rows_ = np.empty((self.B, n), dtype=np.int32)
        self.patch_cols_ = np.empty((self.B, m), dtype=np.int32)
        self.mp_labels_raw_ = np.empty((self.B, n), dtype=np.int32)
        self.mp_labels_ = np.empty((self.B, n), dtype=np.int32)
        self.models_ = []
        self.parallel_MP = n_jobs != 1
        patches = iter_minipatches(N, M, self.B, n, m,
                                  rng=np.random.RandomState(self.random_state))
        with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
            with Parallel(n_jobs=n_jobs, batch_size=1, pre_dispatch="n_jobs") as pool:
                while batch := list(islice(patches, patch_batch_size)):
                    fitted = pool(delayed(_fit_patch)(
                        self.base_clusterer, self.base_classifier, Xs,
                        self.z_ref, rows, cols) for _, rows, cols in batch)
                    for (b, rows, cols), (raw, aligned, model) in zip(batch, fitted):
                        self.patch_rows_[b], self.patch_cols_[b] = rows, cols
                        self.mp_labels_raw_[b], self.mp_labels_[b] = raw, aligned
                        self.models_.append(model)
        self._build_feature_to_patches(M)
        self.X_fit_ = Xs
        return self

    def _score_data(self, X):
        if not hasattr(self, "X_fit_"):
            raise NotFittedError("Run fit() or fit_stream() first")
        Xs = self._prepare_score_data(X)
        if Xs.shape != (self.N_, self.M_):
            raise ValueError("X must have the fitted shape and observation order for LOO scoring")
        return Xs

    def _iter_predictions(self, Xs, proba, patches):
        for b in patches:
            if self.patch_rows_.shape[1] == self.N_:
                rows = np.empty(0, dtype=np.int32)
                pred = np.empty((0, self.K) if proba else (0,),
                                dtype=np.float32 if proba else np.int32)
            else:
                rows, pred = self._mp_predict_omp(Xs, int(b), proba)
            yield int(b), rows, pred

    def iter_patch_predictions(self, X=None, *, proba=True, patches=None):
        """Yield (patch_id, OMP_rows, predictions), lazily, one patch at a time.

        The iterator does not cache results. Callers retaining yielded arrays
        take responsibility for their memory. X uses training observation order.
        """
        Xs = self._score_data(X)
        if patches is None:
            patches = range(self.B)
        for b in patches:
            if not isinstance(b, Integral) or not 0 <= b < self.B:
                raise ValueError("patch indices must be integers in [0, B)")
            yield from self._iter_predictions(Xs, proba, (b,))

    def _stream_predict(self, X, proba):
        Xs = self._score_data(X)
        sums = np.zeros((self.N_, self.K), dtype=np.float32 if proba else np.int32)
        counts = np.zeros(self.N_, dtype=np.int32)
        for _, rows, pred in self._iter_predictions(Xs, proba, range(self.B)):
            _add(sums, counts, rows, pred, proba)
        return _prediction(sums, counts, proba)

    def predict(self, X=None):
        """Stream hard-label OMP votes without creating a prediction cache."""
        return self._stream_predict(X, False)

    def predict_proba(self, X=None):
        """Stream OMP probabilities without creating a prediction cache."""
        return self._stream_predict(X, True)

    def score(self, error_metric=None, z=None, X=None, agg="mean", features=None,
              proba_error=True, parallel_features=False, par=None, *,
              cache="auto", max_cache_bytes=256 * 1024**2, cache_dir=None):
        """Score using one prediction pass and feature-wise cached aggregation.

        Return keys and aggregation semantics match ClusterLOCOMP.score.
        ``max_cache_bytes`` caps the RAM cache payload, not total process RAM.
        Disk mode needs the same payload size in scratch space. Cache resources
        are released on success and exceptions; no cache is reused across calls.
        Feature scoring is intentionally serial; parallel_features=True is
        rejected rather than silently allocating per-worker buffers. ``par`` is
        accepted for API compatibility but unused in serial scoring.
        """
        if parallel_features:
            raise ValueError("Cached scoring is sequential; use parallel_features=False")
        if agg not in {"mean", "none", "by_clusters"}:
            raise ValueError("agg must be 'mean', 'none', or 'by_clusters'")
        Xs = self._score_data(X)
        z = np.asarray(self.z_ref if z is None else z)
        if z.shape != (self.N_,):
            raise ValueError(f"z must have shape ({self.N_},)")
        feats = np.arange(self.M_, dtype=np.int32) if features is None else np.asarray(features)
        if feats.ndim != 1 or (feats.size and (feats.dtype.kind not in "iu" or
                np.any(feats < 0) or np.any(feats >= self.M_))):
            raise ValueError("features must be one-dimensional integer indices in [0, M)")
        feats = feats.astype(np.int32)
        proba = error_metric is not None and bool(proba_error)
        sums = np.zeros((self.N_, self.K), dtype=np.float32 if proba else np.int32)
        counts = np.zeros(self.N_, dtype=np.int32)
        with _PredictionCache(self.B, self.N_ - self.patch_rows_.shape[1],
                              self.K, proba, cache, max_cache_bytes, cache_dir) as saved:
            for b, rows, pred in self._iter_predictions(Xs, proba, range(self.B)):
                saved.put(b, rows, pred)
                _add(sums, counts, rows, pred, proba)
            return self._score_cached(saved, sums, counts, z, feats, error_metric, proba, agg)

    def _score_cached(self, saved, sums, counts, z, feats, metric, proba, agg):
        baseline = _prediction(sums, counts, proba)
        loo_vals = (np.asarray(adjusted_rand_score(z, baseline)) if metric is None
                    else np.asarray(metric(z, baseline)))
        scalar = loo_vals.ndim == 0
        if not scalar and loo_vals.shape != (self.N_,):
            raise ValueError(f"error_metric must return a scalar or shape ({self.N_},)")
        F = len(feats)
        result = dict(metric="adjusted_rand_score" if metric is None else getattr(metric, "__name__", "error_metric"),
                      features=feats, loo_pred=baseline, loo_count=counts, proba=proba)
        delta = np.full(F, np.nan, dtype=np.float32)
        se = np.full(F, np.nan, dtype=np.float32)
        loco_value = np.full(F, np.nan, dtype=np.float32)
        n_eff = np.zeros(F, dtype=np.int32)
        all_diff = np.empty((F, self.N_), dtype=np.float32) if not scalar and agg == "none" else None
        clusters = pd.unique(z) if not scalar and agg == "by_clusters" else None
        by_cluster = np.full((F, len(clusters)), np.nan, dtype=np.float32) if clusters is not None else None
        included = np.zeros_like(sums)
        included_count = np.zeros_like(counts)
        for pos, j in enumerate(feats):
            included.fill(0)
            included_count.fill(0)
            for b in self.feature_to_patches_[j]:
                rows, pred = saved.get(b)
                _add(included, included_count, rows, pred, proba)
            loco_count = counts - included_count
            loco = _prediction(sums - included, loco_count, proba)
            vals = (np.asarray(adjusted_rand_score(z, loco)) if metric is None
                    else np.asarray(metric(z, loco)))
            if vals.shape != loo_vals.shape:
                raise ValueError("error_metric must return the same shape for LOO and LOCO")
            if scalar:
                loco_value[pos] = vals
                delta[pos] = loo_vals - vals if metric is None else vals - loo_vals
                continue
            valid = (counts > 0) & (loco_count > 0) & ~np.isnan(loo_vals) & ~np.isnan(vals)
            diff = np.full(self.N_, np.nan, dtype=np.float32)
            diff[valid] = vals[valid] - loo_vals[valid]
            if all_diff is not None:
                all_diff[pos] = diff
            elif by_cluster is not None:
                for c, label in enumerate(clusters):
                    values = diff[z == label]
                    if np.any(~np.isnan(values)):
                        by_cluster[pos, c] = np.nanmean(values)
            else:
                n_eff[pos] = valid.sum()
                if n_eff[pos]:
                    delta[pos] = np.nanmean(diff[valid])
                    loco_value[pos] = np.nanmean(vals[valid])
                if n_eff[pos] > 1:
                    se[pos] = np.nanstd(diff[valid], ddof=1) / np.sqrt(n_eff[pos])
        if scalar:
            result.update(delta=delta, delta_se=None, loo_value=float(loo_vals), loco_value=loco_value)
        elif all_diff is not None:
            result.update(delta=all_diff, delta_se=None)
        elif by_cluster is not None:
            for pos, values in enumerate(by_cluster):
                valid = ~np.isnan(values)
                if valid.any():
                    delta[pos] = np.nanmean(values)
                if valid.sum() > 1:
                    se[pos] = np.nanstd(values, ddof=1) / np.sqrt(valid.sum())
            result.update(delta=delta, delta_se=se, delta_by_cluster=by_cluster, clusters=clusters)
        else:
            result.update(delta=delta, delta_se=se, loco_value=loco_value, n_eff=n_eff)
        return result
