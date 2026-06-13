"""
Code base for model selection toolsets including data splitting, k-fold cv and subsampling

Author: Claire He

- data_split: wrapper to use in general, pass argument in method to chose k_fold or multiple 
- k_fold_split: base function for k_fold data splitting. When k=n-1 LOO-split
- subsample : multiple random train/test splits 
"""
import numpy as np
from sklearn.base import clone

def data_split(X, method='split', n_split=1, ratio=0.6, shuffle=True, random_state=234, ind=False):
    if method=='split':
        n = X.shape[0]
        if shuffle: 
            rng = np.random.RandomState(random_state)
            indices = np.arange(n)
            rng.shuffle(indices)
            X_train = X[indices[int(ratio*n):]]
            X_test = X[indices[:int(ratio*n)]]
        else:
            X_train = X[int(ratio*n):]
            X_test = X[:int(ratio*n)]
        return [(X_train, X_test)]
    if method == 'k_fold':
        return k_fold_split(X, n_split=n_split, shuffle=shuffle, random_state=random_state, ind=ind)
    elif method == 'subsample':
        return subsample(X, ratio=ratio, shuffle=shuffle, random_state=random_state, ind=ind)
    else:
        print('choose method between k-fold or subsample')
            

def k_fold_split(X, n_split=5, shuffle=True, random_state=None, ind=False):
    """
    True K-fold data splitting: disjoint test sets, each sample used once as test.
    When use k = n-1, LOO-split

    Parameters
    ----------
    X: ndarray
        Dataset of shape (n_samples, n_features)
    n_split: int
        Number of folds (default: 5)
    shuffle: bool
        Whether to shuffle before splitting
    seed: int
        Random seed for reproducibility
    ind: bool
        Whether to return index sets along with the data

    Returns
    -------
    splits: list of (X_train, X_test) or (X_train, X_test, (train_idx, test_idx))
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_split, n_samples // n_split)
    fold_sizes[:n_samples % n_split] += 1

    splits = []
    current = 0

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train = X[train_idx]
        X_test = X[test_idx]

        if ind:
            splits.append((X_train, X_test, (train_idx, test_idx)))
        else:
            splits.append((X_train, X_test))

        current = stop

    return splits
    
def subsample(X, ratio=0.8, shuffle=True, random_state=None, ind=False):
    """ 
    Generate random subsample with given ratio. Data can be redundant.
    Use for Ben-hur stability.

    Parameters
    ----------
    X: ndarray
        data, shape (n_samples, n_features)
    n_split: int
        number of splits, default is 2
    ratio: float
        ratio of subsampling
    shuffle: bool
        whether to shuffle data before each split
    seed: int
        random seed for reproducibility
    ind: bool
        if True, return index sets along with splits

    Returns
    -------
    splits: list of [X_train, X_test] or [X_train, X_test, (train_idx, test_idx)]
    """
    n_samples = len(X)
    rng = np.random.RandomState(random_state)
    n_train = int(ratio*n_samples)
    if shuffle:
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    train_idx = indices[:n_train]
    X_train = X[train_idx]
    
    return (X_train, train_idx) 
    
