from tinygp.kernels.distance import L2Distance
from tinygp.kernels import Kernel, Matern32
import jax
import jax.numpy as jnp
from jax import jit

from tinygp import kernels, transforms, GaussianProcess

jax.config.update("jax_enable_x64", True)

l2_dist = L2Distance()


class Sigmoid2D(Kernel):
    """
    To account for smaller length scales near phase 0.
    Uses generalized logistic curve.
    """
    def __init__(self, log_scale, distance, alpha):
        self.log_scale = jnp.atleast_1d(log_scale)
        self.distance = distance
        self.alpha = alpha
        
    def evaluate(self, X1, X2):
        sig_kernel = 1. / (1. + jnp.exp(-jnp.add(X1, X2) / jnp.exp(self.log_scale) ))**self.alpha
        return sig_kernel[0]
    
    
class SigmoidMatern(Kernel):
    """
    Mixture kernel: sigmoid + Matern32.
    """
    def __init__(self, params, distance):
        self.log_amp = params[0] #params["log_amp"]
        self.log_scale = params[1:4] #jnp.atleast_1d(params["log_scale"])
        self.distance = distance
        self.off_diag = jnp.atleast_1d(params[4]) #jnp.atleast_1d(params["off_diag"])
        self.alpha = params[5] #params["alpha"]
    
        self.k1 = Sigmoid2D(jnp.exp(self.log_scale[0]), self.distance, self.alpha)
        self.k2 = transforms.Cholesky.from_parameters(
            jnp.exp(self.log_scale[1:]),
            self.off_diag,
            Matern32(distance=self.distance)
        )
    def evaluate(self, X1, X2):
        X1_2d, X2_2d = jnp.atleast_1d(X1), jnp.atleast_1d(X2)
        k1 = self.k1.evaluate(X1_2d, X2_2d)

        k2 = self.k2.evaluate(X1_2d, X2_2d)
        
        res = jnp.exp(self.log_amp) * k1 * k2

        del X1_2d, X2_2d, k1, k2
        
        return res
        
        
def build_gp(params, X, yerr):
    kernel = SigmoidMatern(params, l2_dist)
    return GaussianProcess(kernel, X, diag=yerr**2)

@jax.jit
def neg_log_likelihood(theta, X, y, yerr):
    return -build_gp(theta, X, yerr).log_probability(y)

@jax.jit
def interpolate_with_gp(theta, X, y, yerr, X_new):
    gp = build_gp(theta, X, yerr)
    cond_gp = gp.condition(y, X_new).gp
    loc, var = cond_gp.loc, cond_gp.variance
    del gp, cond_gp
    return loc, var
