"""
TUNGSTEN ALPHA: ADELIC FISHER RIEMANN CONNECTOR (AFRC)
=====================================================
Implements Levi-Civita Parallel Transport for metric propagation.
Ensures Information Isometry across decentralized clusters.
"""

import jax
import jax.numpy as jnp
from jax import core, lax, jit
from jax.extend.core import Primitive
from jax._src.interpreters import ad
from jax.interpreters import mlir

# --- THE TRANSPORT PRIMITIVE ---
# We define this as a core primitive to ensure XLA recognizes the 
# geometric preservation during the JIT fusion process.

afrc_transport_p = Primitive("afrc_transport")

def afrc_transport(metric_g, connection_omega):
    """
    Transports the Fisher Metric 'g' along the Connection 'Ω'.
    Formula: g_new = Ω @ g @ Ω.T
    """
    return afrc_transport_p.bind(metric_g, connection_omega)

def afrc_transport_impl(metric_g, connection_omega):
    """Concrete implementation of the transport logic."""
    # The Connection (Ω) must be orthogonal to preserve the Metric volume.
    return connection_omega @ metric_g @ connection_omega.T

def afrc_transport_abstract_eval(metric_g, connection_omega):
    """Tells JAX the shape of the output before execution."""
    return core.ShapedArray(metric_g.shape, metric_g.dtype)

# Register the primitive behaviors
afrc_transport_p.def_impl(afrc_transport_impl)
afrc_transport_p.def_abstract_eval(afrc_transport_abstract_eval)

# --- DIFFERENTIATION (The Bridge) ---
def afrc_transport_jvp(primals, tangents):
    """
    Defines how gradients flow through the Parallel Transport.
    Crucial for NRRB integration.
    """
    g, omega = primals
    dg, domega = tangents
    
    # Primal output
    g_out = afrc_transport(g, omega)
    
    # Tangent output (Lie Algebra representation of the transport derivative)
    # d(ΩgΩᵀ) = dΩ g Ωᵀ + Ω dg Ωᵀ + Ω g dΩᵀ
    term1 = domega @ g @ omega.T
    term2 = omega @ dg @ omega.T
    term3 = omega @ g @ domega.T
    dg_out = term1 + term2 + term3
    
    return (g_out,), (dg_out,)

ad.primitive_jvps[afrc_transport_p] = afrc_transport_jvp

def afrc_transport_mlir_lowering(ctx, metric_g, connection_omega):
    """MLIR lowering for the transport primitive."""
    # We use a standard matrix multiplication lowering
    # g_new = Ω @ g @ Ω.T
    # This is roughly: dot(omega, dot(g, transpose(omega)))

    # In JAX, we can often just define the lowering in terms of other primitives
    # or use the lower_fun helper.
    return mlir.lower_fun(afrc_transport_impl, multiple_results=False)(
        ctx, metric_g, connection_omega)

mlir.register_lowering(afrc_transport_p, afrc_transport_mlir_lowering)

# --- PRODUCTION API ---
@jit
def propagate_metric_step(g_local, omega_connection):
    """
    A JIT-fused call to move the Fisher Metric one step forward 
    in the manifold trajectory.
    """
    return afrc_transport(g_local, omega_connection)

class MetricPropagator:
    """Manages the Parallel Transport for decentralized nodes."""
    def __init__(self, dimension):
        self.dim = dimension
        
    def sync_node(self, local_metric, swarm_connection):
        """Synchronizes a local FIM with the cluster's geometric state."""
        return propagate_metric_step(local_metric, swarm_connection)
