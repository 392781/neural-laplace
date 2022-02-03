from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from kernel import NTK
import matplotlib.pyplot as plt 
import numpy as np

def plot(X, y, typ, title=None):

    try:
        X, X_subset = X
        y, y_subset = y
    except: pass

    ncols = 3 if X.shape[1] == 2 else 1
    figsize = (10, 3)

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
            ax[0].set_ylabel('Kernel Values')
            ax[1].plot(X[:,1], y)
            ax[1].set_xlabel('y')
            ax[2].plot(X[:,0], X[:,1], y)
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
        elif ncols == 1:
            ax.plot(X, y)
            ax.set_xlabel('x')
            ax.set_ylabel('Kernel Values')
    elif typ == 'sample':
        if ncols == 3:
            for i, prior in enumerate(y):
                ax[0].plot(X[:,0], prior, linestyle="--", alpha=0.5, label=f"Sample #{i + 1}")
                ax[1].plot(X[:,1], prior, linestyle="--", alpha=0.5)
                ax[2].plot(X[:,0], X[:,1], prior, alpha=0.5)
        elif ncols == 1:
            for i, prior in enumerate(y):
                ax.plot(X, prior, alpha=0.5, label=f"Sample #{i + 1}")
        
        fig.legend(bbox_to_anchor=(0, .85, 1, 0.2), loc='center', ncol=3)
        
    fig.suptitle(title)

    return fig, ax



def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation