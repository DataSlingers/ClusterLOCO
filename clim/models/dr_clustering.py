"""
Author: Claire HE

To build a compatible object for our feature importance scores, clustering algorithms need to be classes with a `fit`, `predict` and/or `fit_predict` method.

DR_Clustering takes a dim reduction algorithm and a clustering and fits them end to end. 
"""

from sklearn.decomposition import PCA
from sklearn.base import (
    clone,
    BaseEstimator,
    ClusterMixin,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
import inspect

def _set_if_exists(estimator, **kwargs):
    params = estimator.get_params()
    valid = {k: v for k, v in kwargs.items() if k in params}
    if valid:
        estimator.set_params(**valid)
        
# class DR_Clustering(TransformerMixin, ClusterMixin, ClassNamePrefixFeaturesOutMixin):
#     def __init__(self, dim_red_algo, clusterer, n_clusters=None, scaler=None, random_state=42, **kwargs):
#         # Pass dim red algo and clusterer that have already been instantiated


#         self.random_state = random_state
#         self.dim_red_algo = dim_red_algo
#         self.clusterer = clusterer
#         self.scaler = scaler
#         if n_clusters is not None:
#             self.K = int(n_clusters)
#             _set_if_exists(self.clusterer, n_clusters=self.K)
#         _set_if_exists(self.dim_red_algo, random_state=self.random_state)
#         _set_if_exists(self.clusterer, random_state=self.random_state)
        
        
#     def fit(self, X, y=None):
#         if self.scaler is not None:
#             X = self.scaler.fit_transform(X)
#         proj = clone(self.dim_red_algo)
#         X_dim = proj.fit_transform(X)
#         clst = clone(self.clusterer)
#         clst.fit(X_dim)
#         self.model = clst
#         self.labels_ = clst.labels_
#         return self

#     def fit_predict(self, X, y=None):
#         self.fit(X)
#         return self.labels_

#     def predict(self, X, **kwargs):
#         """ just an alias for fit predict, this is not generative """
#         return self.fit_predict(X, **kwargs)

class DR_Clustering(BaseEstimator, ClusterMixin, ClassNamePrefixFeaturesOutMixin):
    def __init__(self, dim_red_algo, clusterer, n_clusters=None, scaler=None, random_state=42):
        self.dim_red_algo = dim_red_algo
        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.scaler = scaler
        self.random_state = random_state

    def fit(self, X, y=None):
        proj = clone(self.dim_red_algo)
        clst = clone(self.clusterer)

        _set_if_exists(proj, random_state=self.random_state)
        _set_if_exists(clst, random_state=self.random_state)

        if self.n_clusters is not None:
            _set_if_exists(clst, n_clusters=int(self.n_clusters))

        if self.scaler is not None:
            scaler = clone(self.scaler)
            X = scaler.fit_transform(X)
            self.scaler_ = scaler

        X_dim = proj.fit_transform(X)
        clst.fit(X_dim)

        self.dim_red_algo_ = proj
        self.clusterer_ = clst
        self.labels_ = clst.labels_
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y=y)
        return self.labels_

    def predict(self, X):
        if not hasattr(self, "dim_red_algo_") or not hasattr(self, "clusterer_"):
            raise AttributeError("This DR_Clustering instance is not fitted yet.")

        if self.scaler is not None:
            if not hasattr(self, "scaler_"):
                raise AttributeError("Scaler was specified but fitted scaler_ is missing.")
            X = self.scaler_.transform(X)

        if not hasattr(self.dim_red_algo_, "transform"):
            raise AttributeError("dim_red_algo does not support transform.")

        if not hasattr(self.clusterer_, "predict"):
            X_dim = self.dim_red_algo_.transform(X)
            return self.clusterer_.fit_predict(X_dim)

        X_dim = self.dim_red_algo_.transform(X)
        return self.clusterer_.predict(X_dim)

    def get_params(self, deep=True):
        params = {
            "dim_red_algo": self.dim_red_algo,
            "clusterer": self.clusterer,
            "n_clusters": self.n_clusters,
            "scaler": self.scaler,
            "random_state": self.random_state,
        }

        if deep:
            for name in ("dim_red_algo", "clusterer", "scaler"):
                obj = params[name]
                if obj is not None and hasattr(obj, "get_params"):
                    nested_params = obj.get_params(deep=True)
                    for key, value in nested_params.items():
                        params[f"{name}__{key}"] = value

        return params

    def set_params(self, **params):
        if not params:
            return self

        valid_top_level = {
            "dim_red_algo",
            "clusterer",
            "n_clusters",
            "scaler",
            "random_state",
        }

        nested_params = {}

        for key, value in params.items():
            if "__" in key:
                name, sub_key = key.split("__", 1)
                if name not in ("dim_red_algo", "clusterer", "scaler"):
                    raise ValueError(
                        f"Invalid parameter {name!r} for {self.__class__.__name__}. "
                        f"Valid nested estimators are: ['dim_red_algo', 'clusterer', 'scaler']"
                    )
                nested_params.setdefault(name, {})[sub_key] = value
            else:
                if key not in valid_top_level:
                    raise ValueError(
                        f"Invalid parameter {key!r} for {self.__class__.__name__}. "
                        f"Valid parameters are: {sorted(valid_top_level)}"
                    )
                setattr(self, key, value)

        for name, sub_params in nested_params.items():
            obj = getattr(self, name)
            if obj is None:
                raise ValueError(
                    f"Cannot set nested parameters for {name!r} because it is None."
                )
            obj.set_params(**sub_params)

        return self