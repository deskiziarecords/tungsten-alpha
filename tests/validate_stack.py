"""
TUNGSTEN ALPHA: FORMAL STACK VALIDATION
=======================================
Verifies HLO Fusion, Metric Positivity, and Bridge Resurgence.
The 'Crystalline' test suite for the 7-layer manifold.
"""

import jax
import jax.numpy as jnp
from core import TungstenRegistry

def validate_hlo_fusion():
    """Checks if the XLA compiler is fusing the transport logic."""
    print("[TEST] VALIDATING HLO FUSION...")
    dim = 16
    stack = TungstenRegistry.get_stack(dim=dim)
    
    # Define a combined operation
    @jax.jit
    def fused_op(g, omega, signal):
        g_t = stack["propagator"].sync_node(g, omega)
        return stack["state_engine"].update_state(signal, signal, g_t)

    # Compile and inspect the HLO
    lowered = fused_op.lower(jnp.eye(dim), jnp.eye(dim), jnp.zeros(dim))
    hlo = lowered.as_text()
    
    # Check for 'fusion' and 'dot' (matrix multiply) optimization
    if "fusion" in hlo.lower():
        print("--> RESULT: SUCCESS. HLO Fusion detected.")
    else:
        print("--> RESULT: WARNING. HLO Fusion not optimized.")

def validate_metric_positivity():
    """Ensures the Fisher Hardener (AFRH) maintains a valid manifold."""
    print("[TEST] VALIDATING METRIC POSITIVITY...")
    dim = 16
    stack = TungstenRegistry.get_stack(dim=dim)
    
    # Generate a random symmetric matrix
    key = jax.random.PRNGKey(0)
    raw_m = jax.random.normal(key, (dim, dim))
    fvp = raw_m @ raw_m.T 
    
    # Harden via AFRH
    g_stable = stack["hardener"].harden_metric_tensor(fvp)
    
    # Check eigenvalues (Must be > 0 for a valid Riemannian Metric)
    eigenvalues = jnp.linalg.eigvalsh(g_stable)
    if jnp.all(eigenvalues > 0):
        print(f"--> RESULT: SUCCESS. Manifold is Positive Definite. Min EV: {jnp.min(eigenvalues):.4f}")
    else:
        print("--> RESULT: FAILED. Manifold Singularity detected.")

def validate_bridge_flow():
    """Verifies that the NRRB allows gradients through the logic switch."""
    print("[TEST] VALIDATING NRRB GRADIENT FLOW...")
    stack = TungstenRegistry.get_stack(dim=1)
    
    # Calculate gradient at the threshold (0.5)
    grad_fn = jax.grad(lambda x: stack["bridge"].apply_bridge(x))
    g_val = grad_fn(0.5)
    
    if g_val > 0:
        print(f"--> RESULT: SUCCESS. Resurgence detected. Gradient at threshold: {g_val:.4f}")
    else:
        print("--> RESULT: FAILED. Gradient is dead at the threshold.")

if __name__ == "__main__":
    print("=== TUNGSTEN ALPHA FORMAL VERIFICATION ===")
    validate_hlo_fusion()
    validate_metric_positivity()
    validate_bridge_flow()
    print("==========================================")
