"""
Montavon NEON 

https://github.com/jacobkauffmann/neon_demo/blob/master/neon.py#L1

"""
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# soft minpooling layer
def smin(X, s, dim=-1):
    return -(1/s)*torch.logsumexp(-s*X, dim=dim) + (1/s)*np.log(X.shape[dim])

# soft maxpooling layer
def smax(X, s, dim=-1):
    return (1/s)*torch.logsumexp(s*X, dim=dim) - (1/s)*np.log(X.shape[dim])

class NeuralizedKMeans(torch.nn.Module):
    def __init__(self, kmeans):
        super().__init__()
        self.n_clusters = kmeans.n_clusters
        self.centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.double)
        K, D = self.centroids.shape

        self.W = torch.empty(K, K - 1, D, dtype=torch.double)
        self.b = torch.empty(K, K - 1, dtype=torch.double)

        for c in range(K):
            for kk in range(K - 1):
                k = kk if kk < c else kk + 1
                self.W[c, kk] = 2 * (self.centroids[c] - self.centroids[k])
                self.b[c, kk] = (
                    self.centroids[k].pow(2).sum() - self.centroids[c].pow(2).sum()
                )

    def h(self, X):
        return torch.einsum('ckd,nd->nck', self.W, X) + self.b

    def forward(self, X, c=None):
        h = self.h(X)
        out = h.min(-1).values
        return out.max(-1).values if c is None else out[:, c]


def beta_heuristic(model, X):
    fc = model(X)
    return 1/fc.mean()
    
def inc(z, eps=1e-9):
    return z + eps * (2 * (z >= 0) - 1)

def neon(model, X, beta=1.0):
    R = torch.zeros_like(X)
    for i in range(X.shape[0]):
        x = X[[i]]
        h = model.h(x)
        out = h.min(-1).values
        c = out.argmax()
        pk = torch.nn.functional.softmin(beta * h[:, c], dim=-1)
        Rk = out[:, c] * pk

        knc = [k for k in range(model.n_clusters) if k != c]
        z = model.W[c] * (x - 0.5 * (model.centroids[[c]] + model.centroids[knc]))
        z = z / inc(z.sum(-1, keepdims=True))
        R[i] = (z * Rk.view(-1, 1)).sum(0)
    return R

def decision(model, X):
    distances = torch.cdist(X, torch.tensor(model.cluster_centers_))**2
    return distances.argmin(-1)

def margins_kmeans(X, model, c=None):
    """
    Calculates min-max-margin for each point and cluster
    
    Parameters
    ----------
    X : ndarray or torch.Tensor of shape (n_samples, n_features)
        Input data.
    model : sklearn.cluster.KMeans
        A fitted KMeans model.
    c : int or None
        If specified, compute margin for cluster c only.
        If None, return max margin across clusters.
        
    Returns
    -------
    margins : torch.Tensor of shape (n_samples,) or (n_samples, n_clusters)
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.double)
    centroids = torch.tensor(model.cluster_centers_, dtype=torch.double)
    n_clusters = centroids.shape[0]
    
    # Compute squared distances to each centroid
    dists = torch.cdist(X, centroids)**2  # shape (n_samples, n_clusters)
    
    margins = []
    for cluster_c in range(n_clusters):
        # Exclude cluster_c from competitors
        other = [k for k in range(n_clusters) if k != cluster_c]
        margin_c = dists[:, other].min(dim=1).values - dists[:, cluster_c]
        margins.append(margin_c)
    
    margins = torch.stack(margins, dim=1)  # shape (n_samples, n_clusters)
    
    if c is None:
        return margins.max(dim=1).values  # same as forward(X)
    else:
        return margins[:, c]  # same as forward(X, c)
        
##### USAGE EXAMPLE
# from sklearn.cluster import KMeans
# from sklearn.metrics.cluster import contingency_matrix
# from scipy.optimize import linear_sum_assignment
# from sklearn.preprocessing import MinMaxScaler
# # sklearn KMeans

# X = MinMaxScaler().fit_transform(X)

# m = KMeans(n_clusters=3, random_state=77)
# y_pred = m.fit_predict(X)
# C = contingency_matrix(labels_true, y_pred)
# _, best_match = linear_sum_assignment(-C.T)
# y_aligned = np.array([best_match[i] for i in y_pred])

# logits = margins_kmeans(X_tensor, m)

# # Initialize neuralized model
# X_tensor = torch.from_numpy(X)
# nm = NeuralizedKMeans(m)

# # Check match
# assert torch.allclose(logits, nm(X_tensor)), "Mismatch in logits"
# # Explain
# R = neon(nm, X_tensor, beta=1.0)



