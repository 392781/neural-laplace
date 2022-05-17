from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import *
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.stats import qmc
from scipy import optimize
from kernel import NTK
import matplotlib.pyplot as plt 
import numpy as np
import pickle

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

    np.random.seed(12589374)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=624562
    )

    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(
        X_norm, y, test_size=0.5, random_state=624562
    )

    y_train_noisy = np.random.normal(y_train, scale=noise)
    y_train_norm_noisy = np.random.normal(y_train_norm, scale=noise)#*0.65)

    data = {
        'orig' : [X, y],
        'orig train' : [X_train, y_train, y_train_noisy],
        'orig test' : [X_test, y_test],
        'norm' : [X_norm, y],
        'norm train' : [X_train_norm, y_train, y_train_norm_noisy],
        'norm test' : [X_test_norm, y_test_norm]
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



def g(ell, gp, mean_ntk, data):
    try:
        gp.set_params(**{'kernel__k2__length_scale': ell})
    except:
        gp.set_params(**{'kernel__k1__k2__length_scale': ell})


    gp.fit(data[0], data[1])
    mean = gp.predict(data[2])
    
    return np.sqrt(np.mean((mean_ntk - mean)**2))



def experiment(data, depth, alpha=1e-5):
    """
    Data format := `[X_train, y_train, X_test, y_test, norm, noise, name]`

    Outputs dictionary containing `dataset`, `means`, `kernel`, 
    `ntk`, `lap`, and `gaus` information
    """
    norm = data[-3]
    noise = data[-2]
    name = data[-1]

    print(f'{name} :\nnorm  = {norm}\nnoise = {noise}\ndepth = {depth}')


    #########################
    # Neural tangent Kernel #
    #########################


    ntk = (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-9, 1e5)) * 
        NTK(depth=depth, c=2, bias=0.1, 
            bias_bounds=(1e-9, 1e5))
    )

    if noise != 0.0:
        ntk += WhiteKernel(
            noise_level=0.15**2, 
            noise_level_bounds=(1e-4, 1e1)
        )

    gp_ntk = GPR(kernel=ntk, alpha=alpha, normalize_y=True,  n_restarts_optimizer=9, random_state=3480795)
    gp_ntk.fit(data[0], data[1])
    mean_ntk = gp_ntk.predict(data[2])

    print(gp_ntk.kernel_)

    if noise != 0.0:
        const_val = gp_ntk.kernel_.get_params()['k1__k1__constant_value']
        noise_lvl = gp_ntk.kernel_.get_params()['k2__noise_level']
        bias = gp_ntk.kernel_.get_params()['k1__k2__bias']
    else :
        const_val = gp_ntk.kernel_.get_params()['k1__constant_value']
        noise_lvl = None
        bias = gp_ntk.kernel_.get_params()['k2__bias']


    #########################
    #     Laplace Kernel    #
    #########################


    lpk = (
        ConstantKernel(
            constant_value=const_val,
            constant_value_bounds='fixed'
        ) *
        Matern(
            nu=1/2,
            length_scale=1,
            length_scale_bounds='fixed'
        ) 
    )

    if noise != 0.0: 
        lpk += WhiteKernel(
            noise_level=noise_lvl,
            noise_level_bounds='fixed'
        )

    gp_lpk = GPR(kernel=lpk, alpha=alpha, normalize_y=True, n_restarts_optimizer=0, random_state=3480795)

    ell_lpk = optimize.minimize_scalar(g, args=(
        gp_lpk, mean_ntk, data), 
        method='bounded', bounds=[1e-4, 1e-3], options={'maxiter': 10000})
    for i in range(-2, 6):
        tmp = optimize.minimize_scalar(g, args=(
            gp_lpk, mean_ntk, data),
            method='bounded', bounds=[1e-4, 10**i], options={'maxiter': 10000})
        if tmp.fun < ell_lpk.fun:
            ell_lpk = tmp

    try:
        gp_lpk.set_params(**{'kernel__k2__length_scale': ell_lpk.x})
    except:
        gp_lpk.set_params(**{'kernel__k1__k2__length_scale': ell_lpk.x})
    gp_lpk.fit(data[0], data[1])
    mean_lpk_opt = gp_lpk.predict(data[2])

    print(gp_lpk.kernel_)


    #########################
    #    Gaussian Kernel    #
    #########################


    gaus = (
        ConstantKernel(
            constant_value=const_val,
            constant_value_bounds='fixed'
        ) *
        Matern(
            nu=np.inf,
            length_scale=1,
            length_scale_bounds='fixed'
        ) 
    )

    if noise != 0.0: 
        gaus += WhiteKernel(
            noise_level=noise_lvl,
            noise_level_bounds='fixed'
        )

    gp_gaus = GPR(kernel=gaus, alpha=alpha, normalize_y=True, n_restarts_optimizer=0, random_state=3480795)

    ell_gaus = optimize.minimize_scalar(g, args=(
        gp_gaus, mean_ntk, data), 
        method='bounded', bounds=[1e-4, 1e-3], options={'maxiter': 10000})
    for i in range(-2, 6):
        tmp = optimize.minimize_scalar(g, args=(
            gp_gaus, mean_ntk, data),
            method='bounded', bounds=[1e-4, 10**i], options={'maxiter': 10000})
        if tmp.fun < ell_gaus.fun:
            ell_gaus = tmp

    try:
        gp_gaus.set_params(**{'kernel__k2__length_scale': ell_gaus.x})
    except:
        gp_gaus.set_params(**{'kernel__k1__k2__length_scale': ell_gaus.x})
    gp_gaus.fit(data[0], data[1])
    mean_gaus_opt = gp_gaus.predict(data[2])

    print(gp_gaus.kernel_)


    #########################
    #         Data          #
    #########################


    exp_data = {}

    exp_data['dataset'] = {
        'name' : name,
        'norm' : norm,
        'noise': noise,
        'test' : [data[2], data[3]],
        'draw' : data[4]
    }
    exp_data['means'] = (mean_ntk.ravel(), mean_lpk_opt.ravel(), mean_gaus_opt.ravel())
    exp_data['kernel'] = {
        'C' : const_val,
        'W' : noise_lvl,
        'ell_lap' : ell_lpk.x,
        'ell_gaus' : ell_gaus.x,
        'depth' : depth,
        'bias' : bias
    }
    exp_data['ntk'] = {
        'pred_rmse' : None,
        'pred_corr' : None,
        'data_rmse' : np.sqrt(np.mean((data[3].ravel() - mean_ntk.ravel())**2)),
        'data_corr' : np.corrcoef((data[3]).ravel(), (mean_ntk).ravel())[0, 1],
        'resi_corr' : None 
    }
    exp_data['lap'] = {
        'pred_rmse' : ell_lpk.fun,
        'pred_corr' : np.corrcoef((mean_ntk).ravel(), (mean_lpk_opt).ravel())[0, 1],
        'data_rmse' : np.sqrt(np.mean((data[3].ravel() - mean_lpk_opt.ravel())**2)),
        'data_corr' : np.corrcoef((data[3]).ravel(), (mean_lpk_opt).ravel())[0, 1],
        'resi_corr' : np.corrcoef((data[3].ravel()-mean_ntk.ravel()), (data[3].ravel() - mean_lpk_opt.ravel()))[0, 1]
    }
    exp_data['gaus'] = {
        'pred_rmse' : ell_gaus.fun,
        'pred_corr' : np.corrcoef((mean_ntk).ravel(), (mean_gaus_opt).ravel())[0, 1],
        'data_rmse' : np.sqrt(np.mean((data[3].ravel() - mean_gaus_opt.ravel())**2)),
        'data_corr' : np.corrcoef((data[3]).ravel(), (mean_gaus_opt).ravel())[0, 1],
        'resi_corr' : np.corrcoef((data[3].ravel()-mean_ntk.ravel()), (data[3].ravel() - mean_gaus_opt.ravel()))[0, 1]
    }

    return exp_data

def save_data(data, name):
    with open(f'{name}.dat', 'wb') as f:
        pickle.dump(data, f)

def load_data(name):
    try:
        with open(f'{name}.dat', 'rb') as f:
            out = pickle.load(f)
    except:
        raise Exception('Load Failure')
    return out

lh = qmc.LatinHypercube(1, seed=23548709)
def sample(low, high, n):
    smpl = lh.random(n)
    smpl = qmc.scale(smpl, low, high)
    smpl = np.sort(smpl.ravel())
    return smpl