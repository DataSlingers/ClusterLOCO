""" Coverage and interval length for split target of inference

Author: Claire He """

import numpy as np
import clim
from clim.utils import hinge_error
from clim.utils.utils import hungarian_align
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from joblib import delayed, Parallel
from clim.data_splitting import Cluster_LOCO_Split
import argparse
from scipy.stats import norm



def sample_from_P(pi, mu, Cov_k, noise_p, rng):
    cluster_memb = rng.choice(len(pi), p=pi)
    X_signal = rng.multivariate_normal(mean=mu[cluster_memb], cov=Cov_k[cluster_memb])
    X_noise = rng.normal(0, 1, size=noise_p)
    return np.concatenate([X_signal, X_noise]), cluster_memb

def generate_dataset(pi, mu, Cov_k, N, noise_p, rng):
    X = np.empty((N, mu.shape[1] + noise_p))
    y = np.empty(N, dtype=int)
    for i in range(N):
        X[i], y[i] = sample_from_P(pi, mu, Cov_k, noise_p, rng)
    return X, y
    

def compute_target_split(pi, mu, Cov_k, N, noise_p, seed, transfer_clf, base_clusterer, error_metric=hinge_error):
    """ 
    Returns target with split
    """
    rng = np.random.default_rng(seed)
    # Data 
    X, y = generate_dataset(pi, mu, Cov_k, N, noise_p = noise_p, rng=rng)
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    c_tr = base_clusterer.fit(X_tr).predict(X_tr)
    c_tr = hungarian_align(y_tr, c_tr) # align the labels
 
    f_tr = clone(transfer_clf).fit(X_tr, c_tr)
    
    # full transfer prediction
    p_pred = f_tr.predict_proba(X_test) 
    
    n, p = X_test.shape
    loco_err = np.zeros((n, p))
    for j in range(p):
        X_trj = np.delete(X_tr, j, axis=1)
        f_trj = clone(transfer_clf).fit(X_trj, c_tr)
        X_testj = np.delete(X_test, j, axis=1)
        p_predj = f_trj.predict_proba(X_testj)
        
        loco_err[:, j] = error_metric(y_test, p_predj) - error_metric(y_test, p_pred)
    return np.mean(loco_err, axis=0)

def fit_cluster_loco(pi, mu, Cov_k, N, noise_p, seed, transfer_clf, base_clusterer, error_metric=hinge_error):
    rng = np.random.default_rng(seed)
    K = len(pi)
    # Data 
    X, y = generate_dataset(pi, mu, Cov_k, N, noise_p = noise_p, rng=rng)
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
    score, se = Cluster_LOCO_Split(X_tr, X_test, model=base_clusterer, seed=seed, clf=RandomForestClassifier(), K=K)
    return score, se
   

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--noise-p", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--B-pop", type=int, default=10000)
    ap.add_argument("--B-est", type=int, default=10000)
    ap.add_argument("--save-mc", type=bool, default=False)
    args = ap.parse_args()

    K = args.K
    p = args.p # number of features
    noise_p = args.noise_p # number of noise features
    alpha = args.alpha
    N = args.N
    seed = args.seed

    rng = np.random.default_rng(seed)
    pi = rng.uniform(0, 1, size=K)
    pi /= pi.sum()
    mu = rng.choice([-alpha, -alpha/2, 0, +alpha/2, alpha], size=(K, p))
    Cov_k = np.array([np.eye(p) for k in range(K)])

    base_clusterer = KMeans(n_clusters=K)
    transfer_clf = RandomForestClassifier()


    seed_rng = np.random.default_rng(seed)
    seeds = seed_rng.integers(0, 2**32 - 1, size=args.B_pop)

    results_pop = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_target_split)(pi, mu, Cov_k, N, noise_p, int(seed),transfer_clf, base_clusterer, error_metric=hinge_error)
        for seed in seeds
    )
    
    Deltas_pop = np.asarray(results_pop)   # shape (B_pop, p_total)   
    Delta_hat = Deltas_pop.mean(axis=0)
    Delta_se = Deltas_pop.std(axis=0, ddof=1) / np.sqrt(args.B_pop)

    seeds = seed_rng.integers(0, 1000000, size=args.B_est)

    clim_mc = Parallel(n_jobs=-1, verbose=10)(
        delayed(fit_cluster_loco)(
            pi, mu, Cov_k, N, noise_p, int(seed),
            transfer_clf, base_clusterer,
            error_metric=hinge_error
        )
        for seed in seeds
    )
    
    # Each has shape (B_est, p_total)
    Delta_hat_reps = np.vstack([score[0] for score in clim_mc])
    Delta_se_reps = np.vstack([se[1] for se in clim_mc])

    if args.save_mc:
        np.save(f'./results_file/cluster_loco_split_target_mc_{args.exp_id}.npy', results_pop)
        np.save(f'./cluster_loco_split_est_mc_{args.exp_id}.npy', clim_mc)

    # Oracle target from the large independent run
    theta_pop = Deltas_pop.mean(axis=0)
    theta_pop_mcse = Deltas_pop.std(axis=0, ddof=1) / np.sqrt(args.B_pop)
    
    # Evaluation-run estimates and their estimated SEs
    Delta_hat_reps = np.vstack([score[0] for score in clim_mc])
    Delta_se_reps = np.vstack([se[1] for se in clim_mc])
    
    R = Delta_hat_reps.shape[0]
    
    z = norm.ppf(0.95)
    
    lower = Delta_hat_reps - z * Delta_se_reps
    upper = Delta_hat_reps + z * Delta_se_reps
    
    covered = (
        (lower <= theta_pop[None, :])
        & (theta_pop[None, :] <= upper)
    )
    
    coverage_by_feature = covered.mean(axis=0)
    coverage_mcse = np.sqrt(
        coverage_by_feature * (1 - coverage_by_feature) / R
    )
    
    avg_length_by_feature = (upper - lower).mean(axis=0)

    outdir = Path("./results_file")
    outdir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        outdir / f"cluster_loco_split_coverage_{args.exp_id}.npz",
        theta_pop=theta_pop,
        theta_pop_mcse=theta_pop_mcse,
        delta_hat_reps=Delta_hat_reps,
        delta_se_reps=Delta_se_reps,
        coverage=coverage_by_feature,
        coverage_mcse=coverage_mcse,
        avg_length=avg_length_by_feature,
    )


if __name__ == "__main__":
    main()
    
    