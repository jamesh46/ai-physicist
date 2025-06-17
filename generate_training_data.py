#!/usr/bin/env python3
"""
Generate training data for physics equation discovery.
Creates multiple trajectories with different parameters and initial conditions.
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from simulate.oscillators import simulate_simple_harmonic, simulate_damped_oscillator
from dataloaders.database_physics_dataloader import PhysicsDataLoader

def generate_sho_dataset(n_trajectories: int = 20):
    """Generate diverse simple harmonic oscillator data."""
    print(f"Generating {n_trajectories} simple harmonic oscillator trajectories...")
    
    trajectory_ids = []
    for i in range(n_trajectories):
        # Random parameters
        k = np.random.uniform(0.5, 3.0)
        m = np.random.uniform(0.5, 2.0) 
        x0 = np.random.uniform(-2.0, 2.0)
        v0 = np.random.uniform(-1.0, 1.0)
        noise = np.random.uniform(0.0, 0.05)
        
        traj_id = simulate_simple_harmonic(
            k=k, m=m, x0=x0, v0=v0, 
            n_steps=1000, dt=0.01, noise=noise
        )
        trajectory_ids.append(traj_id)
    
    return trajectory_ids

def generate_damped_dataset(n_trajectories: int = 20):
    """Generate diverse damped oscillator data."""
    print(f"Generating {n_trajectories} damped oscillator trajectories...")
    
    trajectory_ids = []
    for i in range(n_trajectories):
        # Random parameters
        omega = np.random.uniform(0.5, 2.0)
        zeta = np.random.uniform(0.05, 0.5)  # Underdamped
        x0 = np.random.uniform(-2.0, 2.0)
        v0 = np.random.uniform(-1.0, 1.0)
        noise = np.random.uniform(0.0, 0.05)
        
        traj_id = simulate_damped_oscillator(
            omega=omega, zeta=zeta, x0=x0, v0=v0,
            n_steps=1000, dt=0.01, noise=noise
        )
        trajectory_ids.append(traj_id)
    
    return trajectory_ids

def main():
    print("🎯 Generating Physics Training Dataset")
    print("=" * 40)
    
    # Generate data
    sho_ids = generate_sho_dataset(30)
    damped_ids = generate_damped_dataset(30)
    
    # Check the results
    loader = PhysicsDataLoader()
    print("\n📊 Dataset Summary:")
    loader.print_summary()
    
    # Show sample statistics
    sho_states, sho_derivs = loader.get_training_data("simple_harmonic")
    damped_states, damped_derivs = loader.get_training_data("damped_oscillator")
    
    print(f"\n📈 Data Statistics:")
    print(f"SHO trajectories: {len(sho_ids)}, total points: {len(sho_states)}")
    print(f"Damped trajectories: {len(damped_ids)}, total points: {len(damped_states)}")
    print(f"Position range: [{min(sho_states[:,0].min(), damped_states[:,0].min()):.2f}, {max(sho_states[:,0].max(), damped_states[:,0].max()):.2f}]")
    print(f"Velocity range: [{min(sho_states[:,1].min(), damped_states[:,1].min()):.2f}, {max(sho_states[:,1].max(), damped_states[:,1].max()):.2f}]")
    
    print("\n✅ Dataset generation complete!")
    print("🤖 Ready for AI equation discovery training")

if __name__ == "__main__":
    main()
