"""
TUNGSTEN ALPHA: NEURO-RIEMANNIAN RESURGENCE BRIDGE (NRRB)
========================================================
Enables gradient flow through discrete logical thresholds.
Uses Weierstrass-Gaussian resurgence to bypass the 'Binary Lag'.
"""

import jax
import jax.numpy as jnp
import functools
from jax import jit, custom_jvp

@custom_jvp
def resurgent_threshold(x):
    """
    The Primal Switch: A discrete logical gate.
    In the forward pass, it remains a rigid bit-accurate switch.
    """
    threshold = 0.5
    return jnp.where(x > threshold, 1.0, 0.0)

@resurgent_threshold.defjvp
def resurgent_threshold_jvp(primals, tangents):
    """
    The Resurgence Bridge: The differentiable "Ghost" in the machine.
    Uses a Gaussian kernel to provide a gradient even when the switch is off.
    """
    x, = primals
    dx, = tangents
    
    # Sigma controls the 'Blur' of the bridge—the width of the resurgence.
    # As sigma -> 0, it returns to a hard switch.
    sigma = 0.08 
    
    # The Weierstrass Proxy: A Gaussian gradient centered at the threshold.
    # Formula: d/dx [Heaviside(x-t)] \approx Gaussian(x, t, sigma)
    resurgence_gradient = (1.0 / (sigma * jnp.sqrt(2 * jnp.pi))) * \
                          jnp.exp(-0.5 * ((x - 0.5) / sigma)**2)
    
    # Propagate the gradient back through the manifold
    return resurgent_threshold(x), resurgence_gradient * dx

class ResurgenceBridge:
    """The NRRB Engine: Blurring the line between Logic and Gradient."""
    
    def __init__(self, sigma=0.08):
        self.sigma = sigma

    @functools.partial(jit, static_argnums=(0,))
    def apply_bridge(self, manifold_signal):
        """
        Passes a 16D manifold signal through the Resurgent Bridge.
        Allows the Fisher Metric to 'sense' the logic gates ahead.
        """
        return resurgent_threshold(manifold_signal)

    def measure_resurgence_energy(self, signal):
        """Calculates the 'Vibración'—the gradient potential at the gate."""
        # Use JAX to find the gradient intensity at the current signal point
        return jax.grad(lambda x: jnp.sum(resurgent_threshold(x)))(signal)
