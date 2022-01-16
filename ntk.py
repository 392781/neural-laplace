from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.kernel_approximation import Nystroem
import numpy as np

class NTK(Kernel):
    def __init__(
        self, 
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

    # Need's rewrite -> 2007.01580 uses x,z in R^d not whole matricies
    # sigma_0(X,Z) = X @ Z.T -> usually vector x vector = scaler
    #                        -> but with n x m     
    def __call__(self, X, Z=None, eval_gradient=False):
        aug = False
        X_shape = -1
        Z_shape = -1

        if Z is None:
            Z = X
        else:
            X_shape = X.shape[0]
            Z_shape = Z.shape[0]
            A = np.concatenate((X, Z), axis=0)
            X = A 
            Z = A
            aug = True
            
        Σ_mat = X @ Z.T
        # Σ_mat_dot = np.zeros_like(Σ_mat)
        # K = np.zeros((self.depth+1, X.shape[0], Z.shape[0]))
        # K[0] = Σ_mat + self.bias**2
        K = Σ_mat + self.bias**2

        D = np.sqrt((X @ X.T) * (Z @ Z.T))
        
        # What's a good way to add ~0 noise to the denominator here?
        for dep in range(1, self.depth + 1):
            diag = np.diag(Σ_mat) + 1e-10
            denominator = np.sqrt(np.outer(diag, diag))
            λ = np.clip(Σ_mat / denominator, a_min=-1, a_max=1)
            Σ_mat = (self.c / (2 * np.pi)) * (λ * (np.pi - np.arccos(λ)) + np.sqrt(1 - λ**2)) * denominator
            Σ_mat_dot = (self.c / (2 * np.pi)) * (np.pi - np.arccos(λ))
            # K[dep] = K[dep-1] * Σ_mat_dot + Σ_mat + self.bias**2
            K = K * Σ_mat_dot + Σ_mat + self.bias**2

        if eval_gradient:
            # K_gradient = np.gradient(K) #?
            # return K[self.depth], np.empty((X.shape[0], X.shape[0], 0))
            return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            if aug:
                # return (1/self.depth) * K[self.depth, 0:X_shape, X_shape:(X_shape + Z_shape)]
                return (1/self.depth) * K[0:X_shape, X_shape:(X_shape + Z_shape)]
            else:
                # return (1/self.depth) * K[self.depth]
                return (1/self.depth) * K
        
        # return K

    def diag(self, X):
        return np.einsum("ij,ij->i", X, X)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}(depth={1:d}, c={2:.3f}, bias={3:.3f})".format(
                self.__class__.__name__, self.depth, self.c, self.bias)


z = np.linspace(-2, 2, 100)
x = (z**2 + 1) * np.sin(np.linspace(-np.pi, np.pi, 100))
y = (z**2 + 1) * np.cos(np.linspace(-np.pi, np.pi, 100))

X = np.stack((x, y), axis=1)
z = z.reshape((-1, 1))

X_train, X_test, z_train, z_test = train_test_split(
    X, z, test_size=0.5, random_state=624562)

X_train_norm = normalize(X_train)
z_train_norm = normalize(z_train)

X_norm = normalize(X)
z_norm = normalize(z)

# neural_tangent_kernel = (
#     ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-9, 1e5)) * 
#     NTK(depth=2, c=2, bias=0.1, bias_bounds=(1e-5, 1e5)) #+ 
#     # WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-9, 1e5))
# )
# gp_ntk = GPR(kernel=neural_tangent_kernel, alpha=1e-6, n_restarts_optimizer=9)
# gp_ntk.fit(X_train_norm, z_train_norm)

laplace_kernel = (
    ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-9, 1e10)) * 
    Matern(length_scale_bounds=(2e-2*(1/2), 1/2), nu=1/2) #+
    # WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-9, 1e5))
)
gp_lpk = GPR(kernel=laplace_kernel, alpha=1e-6, n_restarts_optimizer=9)
gp_lpk.fit(X_train_norm, z_train_norm)

print('NTK     : ', gp_ntk.kernel_, '\nLaplace : ', gp_lpk.kernel_)