""" 
Code glossary of cluster validity indices
For an almost exhaustive list, see Todeschini's 68 validity index list. 

Author: Claire He  

Assuming X is of shape (n, p) with n observation, p features.
List of cluster validity indices:
    silhouette, BSS, WSS, ball_hall, banfield_raftery, calinski_harabasz, LSSR, ratkowsky_lance, R_squared, davies_bouldin (DB1, DB2), etc

Other helper functions:
    f_kj (helper for CH index), ext_exclude_self: returns extrema of list excluding given position (for DB)

DBCV implemented from outside implementation to show how to wrap the validation index into the method
"""
import numpy as np 
from types import SimpleNamespace
from sklearn.metrics import silhouette_score
# from clim.dbcv_index import *

# def dbcv(X, labels):
#     DBCV_value = DBCV(X, labels)
#     return SimpleNamespace(value=DBCV_value, _method_='max') 

    
def silhouette(X, labels):
    return SimpleNamespace(value=silhouette_score(X, labels), _method_='max')
    
def BSS(X, labels):
    """ Between cluster sum of squares (BSS) 

    sum_{j=1}^p BSS_j = sum_j sum_k n_k * (c_{kj} 0 b_j)**2 

    """
    bss = 0
    b = X.mean(axis=0)
    K = len(np.unique(labels))
    
    for k in np.unique(labels):
        C_k = X[labels == k]
        n_k = C_k.shape[0]
        if n_k == 0:
            continue
        c_k = C_k.mean(axis=0)
        bss += n_k * np.sum((c_k - b) ** 2)
    return bss
       
def WSS(X, labels):
    """ Trace_W or within cluster sum of squares (WSS) 
    
     Sum_k sum_{i in C_k} ||x_i - c_k||^2 = sum_{j=1}^p sum_i (x_{ij} - c_{kj})**2 = sum_j WSS_j 
    """ 
    wss = 0
    for k in np.unique(labels):
        C_k = X[labels == k]
        if C_k.shape[0] == 0:
            continue
        c_k = C_k.mean(axis=0)
        wss += ((C_k - c_k) ** 2).sum()
    return SimpleNamespace(value=wss, _method_='max')

def ball_hall(X, labels):
    """ Ball-Hall index: average within cluster variance
    
     (1/K) * sum_k (1/n_k) * sum_{i in C_k} ||x_i - c_k||^2 
    """
    wss = 0
    K = len(np.unique(labels))
    for k in np.unique(labels):
        C_k = X[labels == k]
        n_k = C_k.shape[0]
        if n_k == 0:
            continue
        c_k = C_k.mean(axis=0)
        wss += ((C_k - c_k) ** 2).sum()/n_k
    wss._method_ = 'max'
    return SimpleNamespace(value=1/K * wss, _method_='max')

def banfield_raftery(X, labels, eps=1e-6):
    """ Banfield Raftery index 

     sum_k n_k * log( (1/n_k) * sum_{i in C_k} ||x_i - c_k||^2 )
    """
    wss = 0
    K = len(np.unique(labels))
    for k in np.unique(labels):
        C_k = X[labels == k]
        n_k = C_k.shape[0]
        if n_k == 0:
            continue
        c_k = C_k.mean(axis=0)
        wss += n_k * np.log(((C_k - c_k) ** 2).sum()/n_k + eps)
    return SimpleNamespace(value=wss, _method_='max')

def f_kj(X, labels, k, j):
    """
     f_{kj} = I(||c_k-c_j||**2 < sum_{s in I_k} ||x_s - c_k||**2/n_k + sum_{t in I_j} ||x_t - c_j||**2/n_j)
    """
    X_k = X[labels == k]
    X_j = X[labels == j]
    
    n_k = X_k.shape[0]
    n_j = X_j.shape[0]
    if n_k == 0 or n_j == 0:
        return 0  # Avoid divide-by-zero or undefined clusters

    c_k = X_k.mean(axis=0)
    c_j = X_j.mean(axis=0)
    
    dist_sq = np.sum((c_k - c_j)**2)

    var_k = ((X_k - c_k) ** 2).sum() / n_k
    var_j = ((X_j - c_j) ** 2).sum() / n_j

    return int(dist_sq < (var_k + var_j))
    
def calinski_harabasz(X, labels, weighted=False):
    """ Calinski-Harabasz index
    
    BSS/(K-1) * (n-K)/WSS

    or W-CH (2019)

    BSS/(K-1) * 1/ [ WSS/(n-K)+ 1/n *sum_{k=1}^{K-1} sum_{j=k+1}^K f_{kj} ]
    with f_{kj} = I(||c_k-c_j||**2 < sum_{s in I_k} ||x_s - c_k||**2/n_k + sum_{t in I_j} ||x_t - c_j||**2/n_j) 
    """
    K = len(np.unique(labels))
    n = X.shape[0]
    if weighted == False:
        CH = BSS(X, labels) / (K-1) * (n-K) / WSS(X, labels)
        return SimpleNamespace(value=CH, _method_='max')
    else:
        F = 0
        for k in range(1, K-1):
            for j in range(k+1, K):
                F += f_kj(X, labels, k, j)
        CH  = BSS(X, labels)/(K-1) * 1/(WSS(X, labels)/(n-K) + F/n)
        return SimpleNamespace(value=CH, _method_='max')

def LSSR(X, labels):
    """ LSSR Hartigan index

    log(BSS/WSS)"""
    LSSR = np.log(BSS(X, labels)/WSS(X, labels))
    return SimpleNamespace(value=LSSR, _method_='max')

def ratkowsky_lance(X, labels):
    """ RL index

    np.sqrt(1/(Kp) sum_{j=1}^p BSS_j/WSS_j)
    """
    n, p = X.shape
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    b = X.mean(axis=0)

    BSS = 0.0
    TSS = 0.0

    for k in unique_labels:
        X_k = X[labels == k]
        n_k = X_k.shape[0]
        if n_k == 0:
            continue
        c_k = X_k.mean(axis=0)
        BSS += np.sum((c_k - b) ** 2) # shape (p, )
        TSS += ((X_k - b) ** 2).sum(axis=0) # shape (p, )
    valid = TSS > 0
    RL_index = np.sqrt((1 / (K * p)) * np.sum(BSS[valid] / TSS[valid])) 
    return SimpleNamespace(value=RL_index, _method_='max')
    
def R_squared(X, labels):
    R2 = BSS(X, labels)/(BSS(X, labels)+WSS(X, labels))
    return SimpleNamespace(value=R2, _method_='max')

def ext_exclude_self(dists, k, method='min'):
    """
    Given a 1D array of distances, return the index and value of the smallest
    element excluding index k.
    """
    dists_copy = dists.copy()
    if method=='min':
        dists_copy[k] = np.inf
        j = np.argmin(dists_copy)
    elif method =='max':
        dists_copy[k] = -np.inf
        j = np.argmax(dists_copy)        
    return j, dists_copy[j]
    
def davies_bouldin(X, labels, version='max'):
    """ sklearn davies bouldin uses 1979 version """
    from sklearn.metrics import davies_bouldin_score
    if version=='max':
        DB1 = davies_bouldin_score(X, labels)
        return SimpleNamespace(value=DB1, _method_='min')
    elif version =='max-min':
        DB2 = 0.0
        unique_labels = np.unique(labels)
        K = len(unique_labels)
        C = np.stack([X[labels == k].mean(axis=0) for k in unique_labels])
        S = np.array([
            np.linalg.norm(X[labels == k] - C[i], axis=1).mean()
            for i, k in enumerate(unique_labels)])
    
        M = pairwise_distances(centroids)          # shape (K, K)
        for i in range(K):
            # distances of centroid i to others
            dist_vec = M[i]                        # length K
            # S_i + S_j for all j
            var_vec  = S[i] + S                    # length K
    
            _, min_dist   = ext_exclude_self(dist_vec, i, method='min')
            _, max_varsum = ext_exclude_self(var_vec,  i, method='max')
    
            DB2 += max_varsum / min_dist
        DB2 = DB2/K
        return SimpleNamespace(value=DB2, _method_='min')
    else: 
        print("Choose version either max (1979) or max-min (2005)") 


    
    