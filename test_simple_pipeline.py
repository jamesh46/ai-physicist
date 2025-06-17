#!/usr/bin/env python3
"""
Simple test of the streamlined database-only pipeline.
This demonstrates:
1. Data generation (simulation)
2. Data storage (directly to SQLite) 
3. Data loading (for AI training)
"""
import sys
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from simulate.oscillators import simulate_simple_harmonic, simulate_damped_oscillator
from dataloaders.database_physics_dataloader import PhysicsDataLoader

def main():
    print("🚀 Testing Streamlined Physics Data Pipeline")
    print("=" * 50)
    
    # 1. Generate some sample data
    print("\n1. Generating simulation data...")
    
    # Generate simple harmonic oscillator data
    traj_id1 = simulate_simple_harmonic(k=1.0, m=1.0, x0=1.0, v0=0.0, n_steps=500, noise=0.01)
    traj_id2 = simulate_simple_harmonic(k=2.0, m=1.0, x0=0.5, v0=0.5, n_steps=500, noise=0.01)
    
    # Generate damped oscillator data  
    traj_id3 = simulate_damped_oscillator(omega=1.0, zeta=0.1, x0=1.0, v0=0.0, n_steps=500, noise=0.01)
    traj_id4 = simulate_damped_oscillator(omega=1.5, zeta=0.2, x0=0.8, v0=0.2, n_steps=500, noise=0.01)
    
    # 2. Load and inspect the data
    print("\n2. Loading data from database...")
    loader = PhysicsDataLoader()
    
    # Print summary
    loader.print_summary()
    
    # 3. Get training data for AI
    print("\n3. Preparing data for AI training...")
    
    # Get all data
    states, derivatives = loader.get_training_data()
    print(f"Total training samples: {len(states)}")
    print(f"State dimension: {states.shape[1]}")
    print(f"State data shape: {states.shape}")
    print(f"Derivative data shape: {derivatives.shape}")
    
    # Get data for specific system
    sho_states, sho_derivatives = loader.get_training_data("simple_harmonic")
    print(f"\nSimple harmonic oscillator data:")
    print(f"  Samples: {len(sho_states)}")
    print(f"  States range: [{states[:,0].min():.3f}, {states[:,0].max():.3f}]")
    
    damped_states, damped_derivatives = loader.get_training_data("damped_oscillator")
    print(f"\nDamped oscillator data:")
    print(f"  Samples: {len(damped_states)}")
    print(f"  States range: [{damped_states[:,0].min():.3f}, {damped_states[:,0].max():.3f}]")
    
    print("\n✅ Pipeline test complete!")
    print("💡 Data is ready for AI equation discovery models")

if __name__ == "__main__":
    main()
