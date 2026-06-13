"""
Non conformity scores used for Cluster LOCO 

Author: Claire He 

----------------
Non conformity scores for multi-class classification 
    - Hinge error measure
    - Margin error measure
    - Brier score 
    (see @Johansson https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7966105 for details)

Not "valid" non conformity score for multiclass but valid in 0/1 case but used:
    - misclassification error rate (with LOO/LOCO and raw)

Non conformity scores for regression/prediction 
    - L1 score
"""
import numpy as np
from types import SimpleNamespace
from  clim.utils.utils import match_labels

def hinge_error(y_true, probs, proba=True):
    """ Hinge error from probability class 
            1 - P_h(y_i|x_i)
    y_true: true labels {0, ..., K-1}
    probs: probability class predicted from model (0, 1)
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    n = probs.shape[0]
    value = 1.0 - probs[np.arange(n), y_true]

    return SimpleNamespace(
        value=value,
        proba=proba,
        name="hinge_error")


def hamming_distance(y_true, y_pred, proba=False):
    """ Hamming distance (see Lange)
            d(y_pred, y_true) = 1/n * sum_i 1(y_pred != y_true)

        y_true: true labels {0, ..., K-1}
        y_preds: predicted labels (unaligned), {0, ..., K-1}
    """
    # realign
    y_preds_aligned = match_labels(y_preds, y_true)
    d = np.mean(1 * (y_preds_aligned!=y_true))
    return SimpleNamespace(
        value=d,
        proba=proba,
        name='hamming_distance'
    )

def margin_error(y_true, probs, proba=True):
    """ Margin error from probability class
           max_{y neq y_i} P_h(y|x_i) - P_h(y_i|x_i)

    y_true: true labels {0, ..., K-1}
    probs: probability class predicted from model (0, 1)
    """    
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    n = probs.shape[0]
    true_probs = probs[np.arange(n), y_true]

    probs_masked = probs.copy()
    probs_masked[np.arange(n), y_true] = -np.inf
    max_other = probs_masked.max(axis=1)

    value = max_other - true_probs

    return SimpleNamespace(
        value=value,
        proba=proba,
        name="margin_error"
    )



    
def error_rate(y_true, y_pred_loo, y_pred_loco, proba=False):
    """
    LOCO Misclassification rate and variance (for LOO predictions)
    """
    y_true, y_pred_loo, y_pred_loco = np.asarray(y_true), np.asarray(y_pred_loo), np.asarray(y_pred_loco)
    N = y_true.size
    e_loo = (y_true != y_pred_loo).astype(float) 
    e_loco = (y_true != y_pred_loco).astype(float)
    misclass = float((e_loco-e_loo).mean())
    var = float(np.var(e_loco-e_loo, ddof=1) / N)   # jackknife variance since loo preds
    std = float(np.sqrt(var))
    return SimpleNamespace(
        value=misclass,
        std=std,
        proba=proba,
        name="loo_error"
    )


def misclassification_error(y_true, y_pred, proba=False):
    """
    Misclassification error
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    N = y_true.size
    misclass = (y_true != y_pred).astype(float).mean()
    std = (y_true != y_pred).astype(float).std()
    return SimpleNamespace(
        value=misclass,
        std=std,
        proba=proba,
        name="misclassification_error"
    )

def l1_error(y_true, preds, proba=False):
    """ |y_i - f_h(x_i)|
    y_true: true labels
    preds: continuous predicted labels (aligned)
    """
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    value = np.abs(y_true-preds)
    
    return SimpleNamespace(
        value=value,
        proba=proba,
        name="l1_error")

def l2_error(y_true, preds, proba=False):
    """ (y_i - f_h(x_i))
    y_true: true labels
    preds: continuous predicted labels (aligned)
    """
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    value = np.sqrt(np.mean((y_true-preds)**2))
    
    return SimpleNamespace(
        value=value,
        proba=proba,
        name="l2_error")