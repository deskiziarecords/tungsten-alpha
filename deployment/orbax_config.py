"""
TUNGSTEN ALPHA: ORBAX CONFIGURATION
===================================
Manages the lifecycle of the Riemannian Manifold state.
Ensures atomic saves and asynchronous hardening to silicon.
"""

import orbax.checkpoint as ocp
import jax
from pathlib import Path

class TungstenPersistenceManager:
    """The Guardian of the Manifold State."""

    def __init__(self, checkpoint_dir: str = "checkpoints/tungsten_alpha"):
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Configure the CheckpointManager (The 'Temporal Regulator')
        # We keep the last 3 'Crystals' to allow for state-reversion if drift occurs.
        options = ocp.CheckpointManagerOptions(
            max_to_keep=3,
            save_interval_steps=100,
            create=True
        )
        
        # Initialize the Manager with a PyTree handler for our Manifold
        self.mngr = ocp.CheckpointManager(
            self.checkpoint_dir.absolute(),
            ocp.PyTreeCheckpointer(),
            options=options
        )

    def save_manifold(self, step: int, state: dict):
        """
        Asynchronously hardens the current manifold state.
        This allows the 'Velocidad' of the OS to continue while the 
        'Silicio' writes to disk in the background.
        """
        # Save is atomic; it either completes fully or not at all.
        self.mngr.save(step, state)
        # Force a wait to ensure the crystalline lattice is locked
        self.mngr.wait_until_finished()
        print(f"[ORBAX] MANIFOLD HARDENED AT STEP {step}")

    def restore_latest(self) -> dict:
        """
        Restores the 'Clarity' of the system from the latest save.
        """
        latest_step = self.mngr.latest_step()
        if latest_step is not None:
            print(f"[ORBAX] RESTORING MANIFOLD FROM STEP {latest_step}")
            return self.mngr.restore(latest_step)
        else:
            print("[ORBAX] NO CRYSTALLINE STATE FOUND. INITIALIZING FROM VACUUM.")
            return None

def get_config():
    """Returns the standard deployment configuration for JAX-XLA clusters."""
    return {
        "directory": "deploy/tungsten_alpha_v1",
        "async_mode": True,
        "cleanup_tmp_directories": True
    }
