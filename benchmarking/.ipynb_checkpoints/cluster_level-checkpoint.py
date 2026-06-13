"""
Adapted from  NEON https://github.com/jacobkauffmann/neon_demo 
Author: Claire He 
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from clim.cluster_generation import generate_dataset
from clim.sklearn_wrappers import Leiden, SpectralClusteringAffinity
from clim.utils import match_labels
from benchmarking.benchmark_measures import *
from benchmarking.neuralized_kmeans import *
from benchmarking.prototypes import *
from sklearn.base import clone
from itertools import combinations


def LRP_cluster(X, K, random_state=42):
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
    model = KMeans(n_clusters=K, random_state=42)
    y = model.fit_predict(X)
    X_tensor = torch.from_numpy(X)
    logits = margins_kmeans(X_tensor, model)
    nm = NeuralizedKMeans(model)
    R = neon(nm, X_tensor, beta=1.0)
    k_values = np.unique(y)
    FI = np.zeros((len(k_values), X.shape[1]))
    for k in k_values:
        FI[k,:] = R[np.where(y==k),:].mean(axis=1)
    return FI


def c_SHAP_cluster(X, K, m = 2, M=100, baseline="zero", random_state=None, method='perm'):
    """
    Compute SHAP-style approximations of fuzzy membership attributions for all data points.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data
    K: int
        number of clusters 
    m: int
        number of membership degrees 
    M : int
        Number of approx iterations
    baseline : str
        How to fill missing features: 'zero' or 'mean'
    random_state : int or None
        Seed for reproducibility
    method: str
        Default is 'perm' for permutation SHAP, can use 'exact' for exact Shapley if low number of features (slower)
    Returns
    -------
    shap_values : ndarray, shape (n_samples, n_clusters, n_features)
        SHAP attributions for each sample, each cluster, each feature.
    """
    n_samples, n_features = X.shape
    shap_values = np.zeros((K, n_features))
    model = FuzzyKMeans(n_clusters=K, m=m)
    model.fit(X)
    y = model.labels_
    k_values = np.unique(y)
    n_k = np.array([len(y[y == k_values[k]]) for k in range(len(k_values))])
    
    for j in range(K):
        for i in range(n_samples):
            if method=='perm':
                shap_values[j, :] += np.abs(SHAP_fuzzy_per_point(
                    X, model, x_idx=i, cluster_j=j,
                    M=M, baseline=baseline, random_state=random_state
                ))
            elif method=='exact':
                shap_values[j, :] += np.abs(compute_shapley_membership(
                    X, model, x_idx=i, cluster_j=j
                ))
    return 1/K * n_k[:, None] * shap_values




