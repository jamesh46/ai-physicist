#!/usr/bin/env python3
"""
Example of how to use the physics data for AI equation discovery.
This shows how to load the data and prepare it for neural network training.
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from dataloaders.database_physics_dataloader import PhysicsDataLoader

def prepare_data_for_ai():
    """Demonstrate how to prepare physics data for AI training."""
    print("🤖 Preparing Physics Data for AI Equation Discovery")
    print("=" * 55)
    
    # Load data from database
    loader = PhysicsDataLoader()
    
    # Get summary
    print("\n📊 Data Overview:")
    loader.print_summary()
    
    # Get training data for both systems
    print("\n🔬 Loading training data...")
    
    # Simple Harmonic Oscillator data
    sho_states, sho_derivatives = loader.get_training_data("simple_harmonic")
    print(f"SHO data shape: states={sho_states.shape}, derivatives={sho_derivatives.shape}")
    
    # Damped Oscillator data  
    damped_states, damped_derivatives = loader.get_training_data("damped_oscillator")
    print(f"Damped data shape: states={damped_states.shape}, derivatives={damped_derivatives.shape}")
    
    # Combined dataset
    all_states, all_derivatives = loader.get_training_data()
    print(f"Total data shape: states={all_states.shape}, derivatives={all_derivatives.shape}")
    
    print("\n📈 Data Statistics:")
    print(f"Position range: [{all_states[:,0].min():.3f}, {all_states[:,0].max():.3f}]")
    print(f"Velocity range: [{all_states[:,1].min():.3f}, {all_states[:,1].max():.3f}]")
    print(f"Position derivative range: [{all_derivatives[:,0].min():.3f}, {all_derivatives[:,0].max():.3f}]")
    print(f"Velocity derivative range: [{all_derivatives[:,1].min():.3f}, {all_derivatives[:,1].max():.3f}]")
    
    # Data normalization (important for neural networks)
    print("\n🔧 Data Preprocessing:")
    states_mean = all_states.mean(axis=0)
    states_std = all_states.std(axis=0)
    derivatives_mean = all_derivatives.mean(axis=0)
    derivatives_std = all_derivatives.std(axis=0)
    
    print(f"States mean: {states_mean}")
    print(f"States std: {states_std}")
    print(f"Derivatives mean: {derivatives_mean}")
    print(f"Derivatives std: {derivatives_std}")
    
    # Normalize data
    states_normalized = (all_states - states_mean) / states_std
    derivatives_normalized = (all_derivatives - derivatives_mean) / derivatives_std
    
    print(f"Normalized states range: [{states_normalized.min():.3f}, {states_normalized.max():.3f}]")
    print(f"Normalized derivatives range: [{derivatives_normalized.min():.3f}, {derivatives_normalized.max():.3f}]")
    
    # Split into train/validation sets
    print("\n📚 Data Splitting:")
    n_samples = len(all_states)
    train_size = int(0.8 * n_samples)
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    X_train = states_normalized[train_idx]
    y_train = derivatives_normalized[train_idx]
    X_val = states_normalized[val_idx]
    y_val = derivatives_normalized[val_idx]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    print("\n✅ Data is ready for AI training!")
    print("📝 Next steps:")
    print("   1. Build a neural network to learn f(x,v) -> (dx/dt, dv/dt)")
    print("   2. Train the network on this normalized data")
    print("   3. Use symbolic regression to extract equations from the trained model")
    print("   4. Compare discovered equations with known physics laws")
    
    return {
        'X_train': X_train,
        'y_train': y_train, 
        'X_val': X_val,
        'y_val': y_val,
        'states_mean': states_mean,
        'states_std': states_std,
        'derivatives_mean': derivatives_mean,
        'derivatives_std': derivatives_std
    }

if __name__ == "__main__":
    try:
        data = prepare_data_for_ai()
        print(f"\n💾 Data dictionary keys: {list(data.keys())}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
