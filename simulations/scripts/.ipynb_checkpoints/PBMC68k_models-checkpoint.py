import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pickle

sys.path.append("../../")

from benchmarking.benchmark_measures import *
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from clim import ClusterLOCOMP, ClusterLOCO_RAMPART, hinge_error
import anndata as ad
import scanpy as sc
from clim.utils.utils import match_labels
from benchmarking import c_SHAP
from benchmarking.neuralized_kmeans import *


# -----------------------------
# Helpers for alignment
# -----------------------------
def get_label_codes_and_names(labels):
    """
    Convert labels to categorical integer codes and keep name mapping.
    """
    cat = pd.Categorical(labels)
    return cat.codes.astype(int), list(cat.categories)


# -----------------------------
# LRP helper
# -----------------------------
def sample_LRP(X, K, random_state=42):
    """
    Neuralized KMeans / NEON-based LRP.
    """
    X_scaled = MinMaxScaler().fit_transform(X).astype('float64')
    model = KMeans(n_clusters=K, random_state=random_state)
    model.fit(X_scaled)

    X_tensor = torch.from_numpy(X_scaled)
    logits = margins_kmeans(X_tensor, model)
    nm = NeuralizedKMeans(model)
    R = neon(nm, X_tensor, beta=1.0)

    return R.numpy(), model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Path to save results pickle")
    args = parser.parse_args()

    # -----------------------------
    # Load dataset
    # -----------------------------
    adata = sc.datasets.pbmc68k_reduced()

    X = np.asarray(adata.X)
    print("data shape:", X.shape)

    labels = adata.obs["bulk_labels"]
    label_codes, label_names = get_label_codes_and_names(labels)

    K = len(np.unique(label_codes))
    print("K =", K)
    print("label names:", label_names)

    res_dict = {}

    # Hyperparameters
    B = 5000
    B_ramp = 1000
    N, p = X.shape
    model = AgglomerativeClustering(n_clusters=K)
    patch_n, patch_m = 300, 200
    alpha_N, alpha_M = patch_n/N, patch_m/p
    print(alpha_N, alpha_M)
    topk = 200
    par = {
        "n_jobs_features": 8,
        "backend": "loky",
        "prefer": "processes",
        "verbose": True,
    }

    res_dict['data'] = {'K':K, 'topk':topk, 'label_names':label_names, 'label_codes':label_codes, # aligned 
                        'info':{'B':B, 'B_ramp':B_ramp, 'N':N,
                                'M':p, 'patch_n':patch_n, 'patch_m':patch_m,
                                'model':model, 'logs':par}}

    # -----------------------------
    # Cluster LOCOMP
    # -----------------------------
    print("-------- Compute Cluster LOCOMP ---------")

    g = ClusterLOCOMP(
        base_clusterer=model,
        K=K
    )
    g.fit(
        X,
        B=B,
        patch_n=patch_n,
        patch_m=patch_m,
        parallel_MP=False,
        parallel=par,
        standardize=True,
    )
    
    locomp_raw = g.score(hinge_error, z = g.z_ref, agg='by_clusters') # make sure that there are K clusters via z_ref compared to z_test
    locomp_raw['consensus_labels'] = g.z_test # g.z_test
    locomp_raw['reference_labels'] = g.z_ref
    res_dict['cluster_loco'] = locomp_raw
    
    # -----------------------------
    # Cluster LOCO RAMPART
    # -----------------------------   

    print("---- Cluster LOCO RAMPART ----")
    gen_fn = ClusterLOCO_RAMPART(
        base_clusterer=model,
        K=K,
        error_metric=hinge_error,
        parallel_MP=True,
        parallel=par,
        standardize=True,
        z_for_score="z_ref",
        alpha_N = alpha_N,
        alpha_M = alpha_M
    )
    out = RAMPART(
        X,
        generalizability_fn=gen_fn,
        B=B,
        ranking_fn=transform_scores_to_ranking,
        top_k=topk,
        gen_kwargs={},
    )

    rampart_info = out['final_model'].score(hinge_error, agg='by_clusters')
    res_dict['rampart']={}
    res_dict['rampart']['consensus_labels'] = out['final_model'].z_test
    res_dict['rampart']['reference_labels'] = out['final_model'].z_ref
    res_dict['rampart']['info'] = rampart_info
    res_dict['rampart']['all_details'] = out
        
    # -----------------------------
    # LRP
    # -----------------------------
    print("-------- Compute LRP ---------")
    try:
        R, model = sample_LRP(X, K)
    
        lrp_labels_raw = model.labels_
    
        lrp_raw = np.array([
            R[lrp_labels_raw == k].mean(axis=0) if np.any(lrp_labels_raw == k) else np.full(R.shape[1], np.nan)
            for k in range(K)
        ])
    
        res_dict["lrp"] = {
            "raw_delta":lrp_raw,
            "raw_labels":lrp_labels_raw,
        }
    except:
        pass
 
    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "wb") as f:
        pickle.dump(res_dict, f)

    print(f"Saved results to {args.out}")
    print("Done.")



if __name__ == "__main__":
    main()