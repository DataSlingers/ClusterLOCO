import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_4_features_scatter(X, features_list, y, labels, savefig=False, filename = ''):
    
    fig = plt.figure(figsize=(20,3))
    ax = fig.add_subplot(141, projection='3d')
    ax.scatter(X[:,features_list[0]], X[:, features_list[1]], X[:, features_list[3]], c = y, cmap = 'Set1')
    ax.scatter(X[:,features_list[0]], X[:, features_list[1]], X[:, features_list[3]], c = labels, cmap = 'Set1')
    ax.set_xlabel(f'Feature {features_list[0]}')
    ax.set_ylabel(f'Feature {features_list[1]}')
    ax.set_zlabel(f'Feature {features_list[2]}')
    
    ax = fig.add_subplot(142)
    ax.scatter(X[:,features_list[0]], X[:, features_list[1]], c = y, cmap = 'Set1')
    ax.set_xlabel(f'Feature {features_list[0]}')
    ax.set_ylabel(f'Feature {features_list[1]}')
    #ax.axis('equal')
    
    ax = fig.add_subplot(143)
    ax.scatter(X[:,features_list[1]], X[:, features_list[2]], c=y, cmap = 'Set1')
    ax.set_xlabel(f'Feature {features_list[1]}')
    ax.set_ylabel(f'Feature {features_list[2]}')
    #ax.axis('equal')
    
    ax = fig.add_subplot(144)
    ax.scatter(X[:,features_list[2]], X[:, features_list[3]], c=y, cmap = 'Set1')
    ax.set_xlabel(f'Feature {features_list[2]}')
    ax.set_ylabel(f'Feature {features_list[3]}')

    if savefig:
        fig.savefig(filename)
    plt.show()



def plot_cov_heatmap(C, ax=None, *, cmap="coolwarm", title=None, show_cbar=True):
    """
    Square heat-map of a covariance / correlation matrix.

    Parameters
    ----------
    C : (d, d) ndarray  – matrix to display
    ax                 – matplotlib axis (created if None)
    cmap : str         – matplotlib colormap name
    title : str | None – title of the subplot
    show_cbar : bool   – whether to draw the colour-bar
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        C,
        ax=ax,
        cmap=cmap,
        center=0,
        square=True,
        cbar=show_cbar,
        cbar_kws=dict(shrink=0.75),
        linewidths=0.5,
    )
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$i$")
    if title is not None:
        ax.set_title(title)

import numpy as np
import matplotlib.pyplot as plt

def pairplot(X, y=None, feature_names=None, figsize=None, bins=30):
    X = np.asarray(X)
    n, d = X.shape
    if figsize is None:
        figsize=(3*d, 3*d)
        
    if feature_names is None:
        feature_names = [f"$X_{{{i+1}}}$" for i in range(d)]
    
    fig, ax = plt.subplots(d, d, figsize=figsize) #, sharex='col', sharey='row')
    
    for i in range(d):
        for j in range(d):
            if i == j:
                ax[i, j].hist(X[:, i], bins=bins, color="gray", alpha=0.7)
            else:
                if y is not None:
                    ax[i, j].scatter(X[:, j], X[:, i], c=y, s=10, cmap="tab10")
                else:
                    ax[i, j].scatter(X[:, j], X[:, i], s=10)

            # Remove inner tick labels
            if i < d - 1:
                ax[i, j].set_xticklabels([])
            if j > 0:
                ax[i, j].set_yticklabels([])

    # Add outer axis labels
    for j in range(d):
        ax[d-1, j].set_xlabel(feature_names[j])
    for i in range(d):
        ax[i, 0].set_ylabel(feature_names[i])

    plt.tight_layout()
    return fig, ax