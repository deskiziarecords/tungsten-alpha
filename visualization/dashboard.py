"""
TUNGSTEN ALPHA: TELEMETRY DASHBOARD
===================================
Visualizes the Riemannian Manifold and Fisher Information Spectrum.
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

class TungstenDashboard:
    """The Visualization Engine for High-Dimensional Clarity."""

    def __init__(self, dim=16):
        self.dim = dim
        plt.style.use('dark_background') # The 'Silicio' aesthetic

    def plot_fisher_spectrum(self, metric_g):
        """
        Visualizes the eigenvalues of the Fisher Information Matrix.
        A 'Crystalline' system will show a stable, non-collapsing spectrum.
        """
        eigenvalues = jnp.linalg.eigvalsh(metric_g)
        
        plt.figure(figsize=(10, 4))
        plt.bar(range(self.dim), eigenvalues, color='#00ffcc', alpha=0.7)
        plt.title("Fisher Information Spectrum (Manifold Stability)")
        plt.xlabel("Eigen-dimension")
        plt.ylabel("Information Density")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.show()

    def plot_poincare_projection(self, state_manifold):
        """
        Projects the 16D state onto a 2D Poincaré Disk.
        Visualizes the 'Calma' of the recursive state updates.
        """
        # Simple projection: using the first 2 principal components
        # In a full version, we would use a true Hyperbolic mapping.
        x = state_manifold[0]
        y = state_manifold[1]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        circle = plt.Circle((0, 0), 1.0, color='white', fill=False, linestyle='--')
        ax.add_artist(circle)
        
        ax.scatter(x, y, c='#ff00ff', s=100, label='Current State')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title("Manifold State (Poincaré Projection)")
        ax.set_aspect('equal')
        plt.legend()
        plt.show()

def render_telemetry_frame(metric, state):
    """One-shot rendering for the executive summary."""
    dash = TungstenDashboard(dim=metric.shape[0])
    dash.plot_fisher_spectrum(metric)
    dash.plot_poincare_projection(state)

if __name__ == "__main__":
    # Simulated data for the demo
    mock_metric = jnp.diag(jnp.linspace(1.0, 0.1, 16))
    mock_state = jnp.array([0.4, -0.3])
    render_telemetry_frame(mock_metric, mock_state)
