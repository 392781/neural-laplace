from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import *
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from kernel import NTK
import matplotlib.pyplot as plt 
import numpy as np

def processing(*columns: tuple, noise: float=0.15) -> dict:
    """
    Args:
        *columns: packaged list of elements
        noise: how much noise to add to data
    Returns:
        Dictionary containing the following keys: 'orig', 'orig train', 'norm', 'norm train'.
        orig        : (X, y)
        orig train  : (X_train, y_train, y_train_noisy)
        norm        : (X_norm, y)
        norm train  : (X_train_norm, y_train, y_train_noisy)
    """

    if len(columns) == 2:
        X = columns[0]
        y = columns[1]
    else:
        X = np.stack(columns[:-1], axis=1)
        y = columns[-1]

    # temp = np.stack(columns, axis=1)
    # norm = normalize(temp, axis=1)
    # # Sanity check
    # np.testing.assert_array_almost_equal(
    #     np.sqrt(np.diag(norm @ norm.T)), 
    #     np.ones_like(y), 
    #     0.000001)

    # *temp, y_norm = np.split(norm, len(columns), axis=1)
    # X_norm = np.squeeze(np.stack(temp, axis=1))
    y = y.reshape(-1,1)

    X_norm = normalize(X, axis=1)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.5, random_state=624562
    )

    X_train_norm, _, y_train_norm, _ = train_test_split(
        X_norm, y, test_size=0.5, random_state=624562
    )

    y_train_noisy = np.random.normal(y_train, scale=noise)
    y_train_norm_noisy = np.random.normal(y_train_norm, scale=noise)#*0.65)

    data = {
        'orig' : (X, y),
        'orig train' : [X_train, y_train, y_train_noisy],
        'norm' : [X_norm, y],
        'norm train' : [X_train_norm, y_train, y_train_norm_noisy]
    }

    return data



def plot(X, y, typ: str, title: str=None, figsize=(10,3)) -> tuple:

    try:
        X, X_subset = X
        y, y_subset = y
    except: pass

    ncols = 3 if X.shape[1] == 2 else 1
    figsize = figsize

    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    if ncols == 3:
        ax[2].remove()
        ax[2] = fig.add_subplot(1,3,3, projection='3d')

    if typ == 'data':
        if ncols == 3:
            try:
                ax[0].scatter(X_subset[:,0], y_subset, alpha=0.6)
                ax[1].scatter(X_subset[:,1], y_subset, alpha=0.6)
                ax[2].scatter(X_subset[:,0], X_subset[:,1], y_subset)
            except: pass
            ax[0].plot(X[:,0], y)
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('z')
            ax[1].plot(X[:,1], y)
            ax[1].set_xlabel('y')
            ax[2].plot(X[:,0], X[:,1], y)
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
        elif ncols == 1:
            try: 
                ax.scatter(X_subset, y_subset, alpha=0.6)
            except: pass
            ax.plot(X, y)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
    elif typ == 'kernel':
        if ncols == 3:
            ax[0].plot(X[:,0], y)
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('Posterior vals')
            ax[1].plot(X[:,1], y)
            ax[1].set_xlabel('y')
            ax[2].plot(X[:,0], X[:,1], y)
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
        elif ncols == 1:
            ax.plot(X, y)
            ax.set_xlabel('x')
            ax.set_ylabel('Posterior vals')
    elif typ == 'kernel scatter':
        if ncols == 3:
            ax[0].scatter(X[:,0], y)
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('Posterior vals')
            ax[1].scatter(X[:,1], y)
            ax[1].set_xlabel('y')
            ax[2].scatter(X[:,0], X[:,1], y)
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
        elif ncols == 1:
            ax.scatter(X, y)
            ax.set_xlabel('x')
            ax.set_ylabel('Posterior vals')
    elif typ == 'sample':
        if ncols == 3:
            for i, prior in enumerate(y):
                ax[0].plot(X[:,0], prior, linestyle="--", alpha=0.5, label=f"Sample #{i + 1}")
                ax[1].plot(X[:,1], prior, linestyle="--", alpha=0.5)
                ax[2].plot(X[:,0], X[:,1], prior, alpha=0.5)
        elif ncols == 1:
            for i, prior in enumerate(y):
                ax.plot(X, prior, alpha=0.5, label=f"Sample #{i + 1}")
        
        # bbox_to_anchor=(x0, y0, width, height)
        fig.legend(bbox_to_anchor=(0, 0.85, 1, 0.2), loc='center', ncol=3)
    
    y=1
    if typ=='sample':
        y=1.075
    fig.suptitle(title, y=y)

    return fig, ax



def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation