"""
TUNGSTEN ALPHA: CORE MANIFOLD ENGINE
====================================
The centralized registry for Adelic-Riemannian Operating Systems.
Author: J. Roberto Jimenez   - tijuanapaint@gmail.com  - @hipotermiah
"""

# --- EXPOSING THE ARCHITECTURE ---
from .bridge import ResurgenceBridge, resurgent_threshold
from .manifold import StaticManifold, boot_manifold
from .trace import TraceCrystallizer
from .state import RecursiveStateManifold
from .geometry import FisherHardener
from .transport import MetricPropagator, afrc_transport

# --- THE UNIVERSAL REGISTRY ---
class TungstenRegistry:
    """
    The Single Source of Truth for the Swarm.
    Binds all 7 layers into a cohesive 'Vibración'.
    """
    VERSION = "0.1.0-Alpha"
    
    @classmethod
    def get_stack(cls, dim=16):
        """Initializes and returns the complete hardened stack."""
        return {
            "bridge": ResurgenceBridge(),
            "manifold": boot_manifold(target_dim=dim),
            "crystallizer": TraceCrystallizer(target_dim=dim),
            "state_engine": RecursiveStateManifold(state_dim=dim),
            "hardener": FisherHardener(model_fn=None), # To be bound in main.py
            "propagator": MetricPropagator(dimension=dim)
        }

# --- METADATA ---
__all__ = [
    "ResurgenceBridge",
    "StaticManifold",
    "TraceCrystallizer",
    "RecursiveStateManifold",
    "FisherHardener",
    "MetricPropagator",
    "TungstenRegistry"
]

print(f"--- TUNGSTEN ALPHA V.{TungstenRegistry.VERSION} INITIALIZED ---")
