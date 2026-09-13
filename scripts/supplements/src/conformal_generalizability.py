""" 
Modified version of Cluster LOCO Split that supports stochastic labels
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scipy.stats import norm
from scipy.optimize import linear_sum_assignment

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score

from .non_conformity_scores import *


def label_alignment(target, source, n_clusters):
    """Map source labels to target labels by maximum overlap."""
    target = np.asarray(target, dtype=int)
    source = np.asarray(source, dtype=int)
    if target.shape != source.shape:
        raise ValueError("Labels must have the same shape")
    agreement = np.zeros((n_clusters, n_clusters), dtype=int)
    np.add.at(agreement, (source, target), 1)
    source_labels, target_labels = linear_sum_assignment(-agreement)
    mapping = np.empty(n_clusters, dtype=int)
    mapping[source_labels] = target_labels
    return mapping


def number_of_clusters(model):
    """Read the cluster count across sklearn and ClusterLOCO estimators."""
    for attribute in ("n_clusters", "n_components", "K"):
        if hasattr(model, attribute):
            return int(getattr(model, attribute))
    params = model.get_params() if hasattr(model, "get_params") else {}
    if "n_clusters" in params:
        return int(params["n_clusters"])
    raise AttributeError("Clustering model does not expose its cluster count.")

class Conformal_Cluster_LOCO:
    """ 
    Conformal Cluster LOCO: Cluster LOCO Split scores.     
        Parameters
        ----------
        X_tr: training samples 
        X_ca: calibration samples
        model: stochastic clustering algorithm (with `.predict_proba`)
        clf: soft classifier (with `.predict_proba`)
        seed: seed
        error_metric: valid non conformity scores (takes X, y, p)
        n_jobs: number of jobs for parallelization across features
        n_jobs: number of jobs for parallelization across features
    """
    def __init__(self, X_tr, X_ca, model, clf=RandomForestClassifier(), seed=24, error_metric=None, n_jobs=-1):
        self.X_tr = X_tr
        self.X_ca = X_ca
        self.model = model
        self.clf = clf
        self.clf_base = clf
        self.seed = seed
        self.error_metric = error_metric
        self.n_jobs = n_jobs

    def _worker_fit(self, j):
        """ Fit classifier for without feature j """
        X_tr_j = np.delete(self.X_tr, j, axis=1)
        clf_j = clone(self.clf_base)
        clf_j.fit(X_tr_j, self.y_tr)
        assert len(clf_j.classes_) == self.K, "need more samples, classes don't match number of clusters"
        return clf_j
        
    def fit(self, stochastic=True):
        """
        (Stochastic) Cluster LOCO Split scores. Fit clusterers (training) and call (parallelized) classifiers. 
        """
        X_tr = self.X_tr
        model = self.model
        n_tr, p = X_tr.shape
        rng = np.random.default_rng(self.seed)
        
        # Cluster on train and calibration datasets 
        self.train_model = clone(model)
        self.train_model.fit(X_tr)
        self.stochastic=stochastic
        if stochastic:
            self.p_tr = self.train_model.predict_proba(X_tr)
            K = self.p_tr.shape[1]
            self.y_tr = np.array([rng.choice(K, p=self.p_tr[i]) for i in range(X_tr.shape[0])])
        else:
            K = number_of_clusters(self.train_model)
            if hasattr(self.train_model, "labels_"):
                self.y_tr = np.asarray(self.train_model.labels_)
            else:
                self.y_tr = self.train_model.predict(X_tr)
        
        # Fit full classifier on training
        self.clf.fit(X_tr, self.y_tr) 
        self.K = K
        clf_vec = Parallel(n_jobs=self.n_jobs, prefer='processes')(delayed(self._worker_fit)(feature) for feature in range(p))
        self.clf_vec = clf_vec
        return self

        # Alignemnt FIX in `_worker_predict`
    def _worker_predict(self, X_test, y_ca_aligned, j): # Pass aligned labels here
        X_ca_j = np.delete(X_test, j, axis=1)
        n_ca = X_ca_j.shape[0]
        clf_j = self.clf_vec[j]
        prob_ca = clf_j.predict_proba(X_ca_j)
        class_to_col = {label: col for col, label in enumerate(clf_j.classes_)}
        if self.error_metric is None:
            cols = np.array([class_to_col[label] for label in y_ca_aligned])
            errors = 1.0 - prob_ca[np.arange(n_ca), cols]
        else:
            errors = np.asarray([self.error_metric(X_ca_j[i], y_ca_aligned[i], prob_ca[i]) for i in range(n_ca)])
        return errors
    
        
    def predict(self, X_test=None, save=True, per_cluster=True):
        """
        Compute LOCO scores via error metric
            s_tr(X_ca, z_ca (clst); p_ca (clf)) - s_tr(X_ca_j, z_ca (clst); p_ca_j (clf))
        """
        rng = np.random.default_rng(self.seed)
        if X_test is None:
            X_ca = self.X_ca
        else: 
            X_ca = X_test
        test_model = clone(self.model)
    
        if self.stochastic:
            test_model.fit(X_ca)
            p_ca = test_model.predict_proba(X_ca)
            K, n_ca, p = p_ca.shape[1], X_ca.shape[0], X_ca.shape[1]
            self.y_ca = np.array([rng.choice(K, p=p_ca[i]) for i in range(n_ca)])
        else:
            self.y_ca = test_model.fit_predict(X_ca)
            K = number_of_clusters(test_model)
            n_ca, p = X_ca.shape[0], X_ca.shape[1]
            
        # Compute error on calibration set 
        if not hasattr(self.clf, "predict_proba"): # soft classifier
            raise ValueError("Need to a classifier with predict_proba")
        # Error on calibration set
        prob_ca = self.clf.predict_proba(X_ca)
        z_ca = self.clf.predict(X_ca) # align via hard labels
        ca_mapping = label_alignment(z_ca, self.y_ca, K) # align cluster labels to clf labels
        y_ca_aligned = ca_mapping[self.y_ca]
        class_to_col = {label: col for col, label in enumerate(self.clf.classes_)}
        if self.error_metric is None: # defaults to hinge
            cols = np.array([class_to_col[label] for label in y_ca_aligned])
            errors = 1.0 - prob_ca[np.arange(n_ca), cols]
        else:
            
            errors = np.asarray([self.error_metric(X_ca[i], y_ca_aligned[i], prob_ca[i]) for i in range(n_ca)])
        errors_j = np.asarray([self._worker_predict(X_ca, y_ca_aligned, j) for j in range(p)]) # FIX alignment
        if save: 
            self.errors_j = errors_j 
            self.errors = errors

        if per_cluster:
            diff = errors_j - errors[None, :]
            cluster_labels = np.asarray(self.clf.classes_)
            n_clusters = len(cluster_labels) 
            
            res ={'feature':np.arange(p),}
            for cl in cluster_labels: 
                mask = y_ca_aligned == cl
                diff_cl = diff[:, mask]
                cluster_mean = np.mean(diff_cl, axis=1)
                cluster_se = (np.std(diff_cl, axis=1, ddof=1))/np.sqrt(diff_cl.shape[1])
                prefix = f"cluster_{cl}"
                res[f'{prefix}_mean'] = cluster_mean
                res[f'{prefix}_se'] = cluster_se
    
            res['global_mean'] = np.mean(diff, axis=1)
            res['global_se'] = np.std(diff, axis=1, ddof=1)/np.sqrt(X_ca.shape[0])
            res = pd.DataFrame(res)
            self.res = res
            return res['global_mean'], res['global_se']
        else: 
            self.cluster_loco = np.mean(errors_j - errors[None,:], axis=1)
            self.cluster_se = np.std(errors_j - errors, axis=1, ddof=1)/np.sqrt(X_ca.shape[0])
            return self.cluster_loco, self.cluster_se

    def _worker_oracle(self, X_test, y_test, j, clf):
        X_test_j = np.delete(X_test, j, axis=1)
        n_test = X_test.shape[0]
        clf_j = clf[j]
        p_test_j = clf_j.predict_proba(X_test_j)
        class_to_col_j = {label:col for col,label in enumerate(clf_j.classes_)}
        if self.error_metric is None: 
            cols_j = np.array([class_to_col_j[label] for label in y_test])
            return 1.0 - p_test_j[np.arange(n_test), cols_j]
        else:
            return np.asarray([self.error_metric(X_test_j[i], int(y_test[i]), p_test_j[i]) for i in range(n_test)])

    def predict_oracle(self, X_test, y_test, X_al, y_al, save=False, method='stochastic'):
        """
        Compute oracle scores using true labels aligned to the train-model
        label system.
        """
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must contain the same number of observations.")
        if len(X_al) != len(y_al):
            raise ValueError("X_al and y_al must contain the same number of observations.")
        
        if method=='LOCO':
            """
            Computes Delta_j((X_i, y_i)_{i in cI_{tr}}) = 1/N_test sum_{i in cI_{test}} s(y_i, f_j(X_{i,-j)|cD_{tr, -j})) - s(y_i,f(X_i)|cD_{tr})) 
            """
            prob_test = self.clf_loco.predict_proba(X_test)
            K = self.K
        
            # Ensure probability column j corresponds to label j
            if not np.array_equal(self.clf_loco.classes_, np.arange(K)):
                raise ValueError("Classifier probability columns do not correspond to "
                    "labels 0, ..., K-1. At least one sampled training cluster "
                    "may be absent.")
        
            if self.error_metric is None:
                errors = 1.0 - prob_test[np.arange(len(X_test)), y_test]
            else:
                errors = np.asarray([self.error_metric(X_test[i], int(y_test[i]), prob_test[i]) for i in range(X_test.shape[0])])
            errors_j = np.asarray([self._worker_oracle(X_test, y_test, j, self.clf_vec_loco) for j in range(X_test.shape[1])])
            
            if save:
                self.oracle_errors_ = errors
                self.oracle_errors_loco_ = errors_j
            oracle_loco = np.mean(errors_j - errors[None, :], axis=1)
            oracle_se = np.std(errors_j - errors[None, :], axis=1, ddof=1)/np.sqrt(X_test.shape[0])       

        elif (method=='stochastic') or (method=='Cluster-LOCO'):
            """
            if stochastic Cluster-LOCO:
                Computes Delta_j((X_i, z_i)_{i in cI_{tr}}) = 1/N_test sum_{i in cI_{test}} s(y_i, f_j(X_{i,-j)|cD^dagger_{tr, -j})) - s(y_i, f(X_i)|cD^dagger_{tr}))
            if Cluster-LOCO:
                Computes Delta_j((X_i, c_i)_{i in cI_{tr}}) = 1/N_test sum_{i in cI_{test}} s(y_i, f_j(X_{i,-j)|widetilde cD_{tr, -j})) - s(y_i,f(X_i)|widetilde cD_{tr})) 

            """
            # Align the true-label system to the realized label system used to
            # fit the transfer classifiers.  For hard Cluster-LOCO these are
            # the fitted cluster labels; for stochastic Cluster-LOCO these are
            # the posterior labels actually sampled in fit().  The target is
            # conditional on that realized training dataset, so aligning via
            # transfer-classifier predictions or posterior expectations would
            # target a different object.
            K = self.K
            if np.any((y_al < 0) | (y_al >= K)):
                raise ValueError("y_al must contain labels in 0, ..., K-1.")
            
            if np.any((y_test < 0) | (y_test >= K)):
                raise ValueError("y_test must contain labels in 0, ..., K-1.")

            realized_train_labels = np.asarray(self.y_tr)
            if len(y_al) != len(realized_train_labels):
                raise ValueError(
                    "y_al must be paired observation-by-observation with the "
                    "training labels realized in fit()."
                )
            if np.any(
                (realized_train_labels < 0) | (realized_train_labels >= K)
            ):
                raise ValueError(
                    "The realized training labels must lie in 0, ..., K-1."
                )

            # agreement[k, l] is the number of training observations whose
            # true label is k and whose realized hard/stochastic label is l.
            agreement = np.zeros((K, K), dtype=float)
            np.add.at(agreement, (y_al, realized_train_labels), 1.0)
            row_ind, col_ind = linear_sum_assignment(-agreement)

            # True-label index -> realized training-label index.
            true_to_train = np.empty(K, dtype=int)
            true_to_train[row_ind] = col_ind
            y_test_aligned = true_to_train[y_test]

            # Classifier probabilities are in the train-model label system
            prob_test = self.clf.predict_proba(X_test)
        
            # Ensure probability column j corresponds to label j
            if not np.array_equal(self.clf.classes_, np.arange(K)):
                raise ValueError(
                    "Classifier probability columns do not correspond to "
                    "labels 0, ..., K-1. At least one sampled training cluster "
                    "may be absent.")
        
            if self.error_metric is None:
                errors = 1.0 - prob_test[np.arange(len(X_test)), y_test_aligned]
            else:
                errors = np.asarray([self.error_metric(X_test[i], int(y_test_aligned[i]), prob_test[i]) for i in range(X_test.shape[0])])
            errors_j = np.asarray([self._worker_oracle(X_test, y_test_aligned, j, self.clf_vec) for j in range(X_test.shape[1])])
            
            if save:
                self.oracle_alignment_ = true_to_train
                self.y_test_aligned_ = y_test_aligned
                self.oracle_errors_ = errors
                self.oracle_errors_loco_ = errors_j
            oracle_loco = np.mean(errors_j - errors[None, :], axis=1)
            oracle_se = np.std(errors_j - errors[None, :], axis=1, ddof=1)/np.sqrt(X_test.shape[0])
            
        return oracle_loco, oracle_se

    def validate(self, X_test, y_test, X_al, y_al, X_ca, y_ca, alpha=0.1, stochastic=True, method='stochastic'):
        self.fit(stochastic=stochastic)
        cluster_loco, cluster_se = self.predict(X_ca)
        self.fit_loco(y_al)
        loco, loco_se = self.predict_loco(X_ca, y_ca)
        oracle_loco, oracle_se = self.predict_oracle(X_test=X_test, y_test=y_test, X_al=X_al, y_al=y_al, method=method)

        z = norm.ppf(1.0 - alpha / 2.0)
        p = X_test.shape[1]
        lower, upper, covered, length = np.zeros((p,)), np.zeros((p,)), np.zeros((p,)), np.zeros((p,))
        loco_low, loco_up, cover_loco, length_loco = np.zeros((p,)), np.zeros((p,)), np.zeros((p,)), np.zeros((p,))
        for j in range(p):
            lower[j] = cluster_loco[j] - z * cluster_se[j]
            upper[j] = cluster_loco[j] + z * cluster_se[j]
            covered[j] = bool(lower[j] <= oracle_loco[j] <= upper[j])
            length[j] = float(upper[j] - lower[j])

            loco_low[j] = loco[j] - z * loco_se[j]
            loco_up[j] = loco[j] + z * loco_se[j]
            cover_loco[j] = bool(loco_low[j] <= oracle_loco[j] <= loco_up[j])
            length_loco[j] = float(loco_up[j] - loco_low[j])
    
        return {
            "feature": np.arange(len(cluster_loco)),
            "covered": covered,
            "cluster_mean": cluster_loco,
            "cluster_se": cluster_se,
            "oracle_loco": oracle_loco,
            "oracle_se": oracle_se,
            "lower": lower,
            "upper": upper,
            "length": length,
            "loco_mean": loco,
            "loco_se":loco_se,
            "loco_cov":cover_loco,
            "loco_low": loco_low,
            "loco_up": loco_up,
            "loco_length": length_loco,
        }

    def _worker_fit_loco(self, j):
        X_tr_j = np.delete(self.X_tr, j, axis=1)
        clf_j = clone(self.clf_base)
        clf_j.fit(X_tr_j, self.y_tr_loco)
        return clf_j
        
    def fit_loco(self, y_tr_loco):
        """
        Supervised LOCO Split scores.  
        """
        X_tr = self.X_tr
        self.y_tr_loco = y_tr_loco
        n_tr, p = X_tr.shape[0], X_tr.shape[1]
        rng = np.random.default_rng(self.seed)
        # Fit full classifier on training
        self.clf_loco = clone(self.clf_base)
        self.clf_loco.fit(X_tr, y_tr_loco) 
        clf_vec = Parallel(n_jobs=self.n_jobs, prefer='processes')(delayed(self._worker_fit_loco)(feature) for feature in range(p))
        self.clf_vec_loco = clf_vec
        return self
    
    def predict_loco(self, X_test=None, y_ca=None):
        if X_test is None:
            X_ca = self.X_ca
        else:
            X_ca = X_test
        y_ca = np.asarray(y_ca)
        n_ca, p = X_ca.shape
        p_ca = self.clf_loco.predict_proba(X_ca)
        class_to_col = {label: col for col, label in enumerate(self.clf_loco.classes_)}
        if self.error_metric is None:
            cols = np.array([class_to_col[label] for label in y_ca])
            errors = 1.0 - p_ca[np.arange(n_ca), cols]
        else:
            errors = np.asarray([self.error_metric(X_ca[i],y_ca[i],p_ca[i]) for i in range(n_ca)])
        errors_j = []
        for j in range(p):
            X_ca_j = np.delete(X_ca, j, axis=1)
            clf_j = self.clf_vec_loco[j]
            p_ca_j = clf_j.predict_proba(X_ca_j)
            class_to_col_j = {label: col for col, label in enumerate(clf_j.classes_)}
            if self.error_metric is None:
                cols_j = np.array([class_to_col_j[label] for label in y_ca])
                errors_j.append(1.0 - p_ca_j[np.arange(n_ca), cols_j])
            else:
                errors_j.append(np.asarray([self.error_metric(X_ca_j[i], y_ca[i], p_ca_j[i]) for i in range(n_ca)]))
        errors_j = np.asarray(errors_j)
        loco_obs = errors_j - errors[None, :]
        loco = np.mean(loco_obs, axis=1)
        loco_se = np.std(loco_obs, axis=1, ddof=1) / np.sqrt(n_ca)
        return loco, loco_se
