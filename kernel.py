import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

class NTK(Kernel):
    def __init__(self, 
        depth, 
        c = 2.0,
        bias = 0.1,
        bias_bounds = (1e-5, 1e5)
    ):
        self.depth = depth
        self.c = c
        self.bias = bias
        self.bias_bounds = bias_bounds

    @property
    def hyperparameter_bias(self):
        return Hyperparameter("bias", "numeric", self.bias_bounds)

    def __call__(self, X, Z=None, eval_gradient=False):
        aug = False
        X_shape = -1
        Z_shape = -1
        products = []

        if Z is None:
            Z = X
        else:
            X_shape = X.shape[0]
            Z_shape = Z.shape[0]
            A = np.concatenate((X, Z), axis=0)
            X = A 
            Z = A
            aug = True

        if eval_gradient:
            products.append(np.ones((X.shape[0], X.shape[0])))
            
        Σ_mat = X @ Z.T

        K = Σ_mat + self.bias**2 # K^0/

        for dep in range(1, self.depth + 1): # K^1 to K^L
            diag = np.diag(Σ_mat)
            denominator = np.clip(np.sqrt(np.outer(diag, diag)), a_min=1e-10, a_max=None)
            div = Σ_mat / denominator
            # div = np.nan_to_num(div)
            λ = np.clip(div, a_min=-1, a_max=1)
            Σ_mat = (self.c / (2 * np.pi)) * (λ * (np.pi - np.arccos(λ)) + np.sqrt(1 - λ**2)) * denominator
            Σ_mat_dot = (self.c / (2 * np.pi)) * (np.pi - np.arccos(λ))
            K = K * Σ_mat_dot + Σ_mat + self.bias**2

            if eval_gradient:
                products.append(products[-1] * Σ_mat_dot)

        scalar = 1/((self.depth + 1) * (self.bias**2 + 1))
            
        if eval_gradient:
            if not self.hyperparameter_bias.fixed:
                K_prime = 2 * self.bias**2 * products[-1] * (1 + sum(1/np.array(products)))
                K_prime = np.expand_dims(K_prime, -1)
                return scalar * K, scalar * (K_prime - ((2 * self.bias**2) / (self.bias**2 + 1)) * np.expand_dims(K, -1))
            else:
                return scalar * K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            if aug:
                return scalar * K[0:X_shape, X_shape:(X_shape + Z_shape)]
            else:
                return scalar * K
        
    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}(depth={1:d}, c={2:.3f}, bias={3:.3f})".format(
                self.__class__.__name__, self.depth, self.c, self.bias)