""" Cluster LOCO Split

Implements Split Cluster LOCO with given train/calibration sets for: 
- negative ARI
- non conformity scores in clim.non_conformity_scores 

Version for reproducibility and experiments.
Author: Claire He 
"""
import numpy as np
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score

from clim.utils.utils import *

def _configure_cluster_count(model, K, count_param=None):
    """Clone a clustering model and resolve its cluster-count parameter."""
    model = clone(model)
    params = model.get_params(deep=False)

    if count_param is None:
        if "n_clusters" in params:
            count_param = "n_clusters"
        elif "n_components" in params:
            count_param = "n_components"
        else:
            raise ValueError(f"{type(model).__name__} exposes neither 'n_clusters' nor 'n_components'.")

    if count_param not in params:
        raise ValueError(f"{type(model).__name__} has no parameter {count_param!r}.")

    if K is None:
        K = params[count_param]
    else:
        model.set_params(**{count_param: K})

    if K is None:
        raise ValueError(f"{type(model).__name__}.{count_param} is None; provide K explicitly.")
    return model, K


def Cluster_LOCO_Split(X_tr, X_ca, model=KMeans(), clf = RandomForestClassifier(), 
                       K=None, seed=24, error_metric=None,use_proba=True, n_jobs=-1, cluster_count_param=None):
    n_tr, p = X_tr.shape
    n_ca, _ = X_ca.shape
    
    np.random.seed(seed)
    model, K = _configure_cluster_count(model, K, cluster_count_param)
    train_model, test_model = clone(model), clone(model)
  
    # Cluster training data and test data
    try:
        y_tr = train_model.fit_predict(X_tr)
    except:
        train_model.fit(X_tr)
        y_tr = train_model.predict(X_tr)

    try:
        y_ca = test_model.fit_predict(X_ca)
    except:  
        test_model.fit(X_ca)
        y_ca = test_model.predict(X_ca)

    # Fit classifier on training
    transfer_clf = clone(clf)
    transfer_clf.fit(X_tr, y_tr)
    # Predict with hard classifier
    z_ca = transfer_clf.predict(X_ca) 
    
    # Compute error on calibration set 
    if hasattr(transfer_clf, "predict_proba"): # soft classifier
        # Error on calibration set
        prob_ca = transfer_clf.predict_proba(X_ca)
        # align via hard labels
        
        ca_mapping = label_alignment(z_ca, y_ca, K) # align cluster labels to classifier labels
        y_ca_aligned = ca_mapping[y_ca]
        
        if error_metric==None: # defaults to hinge
            errors = 1.0 - prob_ca[np.arange(n_ca), y_ca_aligned]
        elif error_metric=='ARI': 
            errors = - adjusted_rand_score(y_ca, z_ca) # agnostic to alignment 
        elif use_proba:
            errors = error_metric(y_ca_aligned, prob_ca)
        else:
            errors = error_metric(z_ca, y_ca_aligned)
    else:
        errors = - adjusted_rand_score(y_ca, z_ca) # agnostic to alignment 

    errors_j = Parallel(n_jobs=n_jobs, prefer='processes')(delayed(compute_loco_error)(X_tr, X_ca, model, K, error_metric, clf, feature, use_proba) for feature in range(p))
    errors_j = np.asarray(errors_j).T
    
    if error_metric == 'ARI':
        cluster_loco = errors_j - errors
        cluster_std = np.zeros(cluster_loco.shape)
    else: 
        errors = errors.reshape(-1, 1)
        cluster_loco = np.mean(errors_j - errors, axis=0)
        cluster_std = np.std(errors_j - errors, axis=0)

    return cluster_loco, cluster_std
    

def compute_loco_error(X_tr, X_ca, model, K, error_metric, clf, feature, use_proba=True):
    n_tr, p = X_tr.shape
    n_ca, _ = X_ca.shape
    
    train_model, test_model = clone(model), clone(model)
  
    # Cluster training data and test data
    try: 
        y_tr = train_model.fit_predict(X_tr)
        y_ca = test_model.fit_predict(X_ca)
    except:    
        train_model.fit(X_tr)
        y_tr = train_model.predict(X_tr)
        
        test_model.fit(X_ca)
        y_ca = test_model.predict(X_ca) 

    # Fit classifier on training
    clf_j = clone(clf)
    # Remove covariate feature
    X_tr_j = np.delete(X_tr, feature, axis=1)
    X_ca_j = np.delete(X_ca, feature, axis=1)
    clf_j.fit(X_tr_j, y_tr) 
    assert len(clf_j.classes_) == K, "need more samples, classes don't match cluster number" 
    # Predict hard classifier
    z_ca = clf_j.predict(X_ca_j)
    # Compute error on calibration set 
    if hasattr(clf_j, "predict_proba"): # soft classifier
        # Error on calibration set
        prob_ca = clf_j.predict_proba(X_ca_j)
        # align via hard labels
        ca_mapping = label_alignment(z_ca, y_ca, K) # align cluster labels to classifier labels
        y_ca_aligned = ca_mapping[y_ca]
        if error_metric==None:
            errors = 1.0 - prob_ca[np.arange(n_ca), y_ca_aligned]
        elif error_metric=='ARI': 
            errors = - adjusted_rand_score(y_ca, z_ca) # agnostic to alignment 
        elif use_proba:
            errors = error_metric(y_ca_aligned, prob_ca)
        else:
            errors = error_metric(z_ca, y_ca_aligned)
    else: 
        errors = - adjusted_rand_score(y_ca, z_ca) # agnostic to alignment 
    return errors
