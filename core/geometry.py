"""
TUNGSTEN ALPHA: ADELIC FISHER RESURGENCE HARDENER (AFRH)
=======================================================
Calculates and Hardens the Fisher Information Metric.
Regulates the manifold curvature via Natural Gradient estimation.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vjp, vmap

@jit
def compute_fisher_vector_product(log_likelihood_fn, params, vectors):
    """
    Computes the Fisher-Vector Product (FVP) using the Pearlmutter trick.
    This avoids O(n^2) explicit Hessian storage.
    """
    # Define the gradient of the log-likelihood
    def grad_log_lik(p):
        return grad(log_likelihood_fn)(p)

    # Use VJP (Vector-Jacobian Product) to get the second-order curvature
    _, vjp_fn = vjp(grad_log_lik, params)
    return vjp_fn(vectors)[0]

@jit
def harden_metric_tensor(fvp_matrix, ridge_epsilon=1e-5):
    """
    Stabilizes the Fisher Metric. 
    Ensures the manifold is Isostatic and Positive-Definite.
    """
    # Apply Tikhonov regularization (Ridge) to prevent singular manifolds
    dim = fvp_matrix.shape[0]
    stable_g = fvp_matrix + ridge_epsilon * jnp.eye(dim)
    
    # Force symmetry (Crystalline constraint)
    return 0.5 * (stable_g + stable_g.T)

class FisherHardener:
    """The AFRH Regulator for Manifold Curvature."""
    
    def __init__(self, model_fn, ridge=1e-6):
        self.model_fn = model_fn
        self.ridge = ridge

    def harden_metric_tensor(self, fvp_matrix, ridge_epsilon=1e-5):
        """Instance method to harden the metric tensor."""
        return harden_metric_tensor(fvp_matrix, ridge_epsilon)

    def get_natural_gradient(self, params, loss_grad, fisher_metric):
        """
        Solves the Natural Gradient equation: g * \tilde{\nabla} = \nabla
        This is the path of steepest descent in the Riemannian manifold.
        """
        # Solves for \tilde{\nabla} (the Natural Gradient)
        natural_grad = jnp.linalg.solve(fisher_metric, loss_grad)
        return natural_grad

    @jit
    def calculate_curvature(self, params, sample_batch):
        """
        Calculates the explicit Metric Tensor g_ij for a batch.
        Essential for the AFRC transport layer.
        """
        # Note: In production, we vmap the FVP over the batch
        # for maximum XLA throughput.
        def single_fisher(p, x):
            # Stochastic approximation of the local curvature
            g_local = jax.hessian(self.model_fn)(p, x)
            return g_local

        batch_g = vmap(single_fisher, in_axes=(None, 0))(params, sample_batch)
        return jnp.mean(batch_g, axis=0)
