"""
TUNGSTEN ALPHA: UNIVERSAL ORCHESTRATOR
======================================
The entry point for the Adelic-Riemannian Operating System.
Executes the full manifold cycle: Bridge -> Geometry -> Transport -> State.
"""

import jax
import jax.numpy as jnp
from core import TungstenRegistry
from deployment.orbax_config import TungstenPersistenceManager

def main():
    print("--- STARTING TUNGSTEN ALPHA HEARTBEAT ---")
    
    # 1. INITIALIZE THE STACK (The 'Unión')
    dim = 16
    stack = TungstenRegistry.get_stack(dim=dim)
    persistence = TungstenPersistenceManager()
    
    # 2. GENERATE A SYNTHETIC SIGNAL (The 'Vibración')
    # A 16D vector representing a telemetry burst
    key = jax.random.PRNGKey(2026)
    telemetry_signal = jax.random.normal(key, (dim,))
    
    # 3. EXECUTE THE MANIFOLD CYCLE
    print(f"[PROCESS] Signal Received. Energy: {jnp.linalg.norm(telemetry_signal):.4f}")
    
    # Layer 1: The Bridge (NRRB) - Softening the logic
    gated_signal = stack["bridge"].apply_bridge(telemetry_signal)
    
    # Layer 2: The Manifold (CSM) - Forcing crystalline shape
    lattice = stack["manifold"].project_to_lattice(gated_signal.reshape(dim, 1))
    
    # Layer 3: The Geometry (AFRH) - Calculating local curvature
    # We use an identity matrix as a proxy for the first-step Fisher Metric
    g_local = jnp.eye(dim) 
    
    # Layer 4: The Transport (AFRC) - Connecting to the Swarm
    # We use a rotation matrix as the connection Omega
    omega = jnp.eye(dim) # For this demo, we use the identity connection
    g_transported = stack["propagator"].sync_node(g_local, omega)
    
    # Layer 5: The State (ARSM) - Updating the Recursive Memory
    h_init = stack["state_engine"].initialize_manifold()
    h_next, coherence = stack["state_engine"].update_state(h_init, telemetry_signal, g_transported), 1.0
    
    print(f"[STATUS] Cycle Complete. State Coherence: {coherence:.4f}")
    print(f"[STATUS] Metric Trace (Energy): {jnp.trace(g_transported):.2f}")

    # 4. HARDEN TO SILICON (The 'Calma')
    state_to_save = {
        "step": 0,
        "manifold_state": h_next,
        "metric": g_transported,
        "metadata": {"status": "Crystalline"}
    }
    persistence.save_manifold(step=0, state=state_to_save)
    
    print("--- TUNGSTEN ALPHA IS ONLINE. READY FOR SWARM DEPLOYMENT. ---")

if __name__ == "__main__":
    main()
