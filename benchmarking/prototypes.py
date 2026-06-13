""" 
Fuzzy KMeans for c_SHAP (Napoles) 

Author: Claire He
""" 

import numpy as np
import pylab as pl

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans 
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

class FuzzyKMeans(BaseEstimator, ClusterMixin):
    """
    Fuzzy K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.

    m : float, default=2
        Fuzziness parameter. Must satisfy m > 1.
        Values close to 1 approach hard k-means. Larger values produce
        softer memberships.

    max_iter : int, default=100
        Maximum number of iterations.

    random_state : int, RandomState instance, or None, default=0
        Random seed.

    tol : float, default=1e-4
        Convergence tolerance based on center movement.
    """

    def __init__(self, n_clusters, m=2, max_iter=100, random_state=0, tol=1e-4):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def _e_step(self, X):
        dist = euclidean_distances(X, self.cluster_centers_, squared=True)
        dist = np.maximum(dist, np.finfo(float).eps)

        memberships = dist ** (-1.0 / (self.m - 1.0))
        memberships /= memberships.sum(axis=1, keepdims=True)

        self.fuzzy_labels_ = memberships
        self.labels_ = memberships.argmax(axis=1).astype(np.int32)

    def _m_step(self, X):
        weights = self.fuzzy_labels_ ** self.m
        denom = weights.sum(axis=0)[:, None]
        self.cluster_centers_ = (weights.T @ X) / denom

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)

        if self.m <= 1:
            raise ValueError("m must be > 1.")

        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)

        self.fuzzy_labels_ = rng.rand(n_samples, self.n_clusters)
        self.fuzzy_labels_ /= self.fuzzy_labels_.sum(axis=1, keepdims=True)

        self._m_step(X)

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            center_shift = np.linalg.norm(self.cluster_centers_ - centers_old)

            if center_shift < self.tol:
                break

        self.n_iter_ = i + 1
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        dist = euclidean_distances(X, self.cluster_centers_, squared=True)
        return np.argmin(dist, axis=1).astype(np.int32)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        dist = euclidean_distances(X, self.cluster_centers_, squared=True)
        dist = np.maximum(dist, np.finfo(float).eps)

        memberships = dist ** (-1.0 / (self.m - 1.0))
        memberships /= memberships.sum(axis=1, keepdims=True)

        return memberships