"""
Create experiment configurations. 
    python experiments/create_config.py 
"""
from pathlib import Path
import numpy as np


output = Path(f"{args.output_folder}/experiments/{args.method}_config.npz")
output.parent.mkdir(parents=True, exist_ok=True)
""" GMM configuration """
if args.method == 'gmm':
    K = 3
    means = np.array([[1, 1], [3, 4], [4, 1]])
    base_covariances = np.array([np.eye(2) for _ in range(K)])
    weights = np.ones(K) / K
    np.savez(output, means=means, base_covariances=base_covariances, weights=weights)

""" Spectral toy example configuration """
elif args.method == 'spectral':


print(f"Saved {output}")
