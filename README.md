# AI Physicist - Streamlined Physics Simulation & Equation Discovery

A simple, database-driven pipeline for generating physics simulation data and training AI models to discover equations.

## Features

✅ **Pure SQLite Storage** - No external files, everything in one database  
✅ **Simple API** - Generate, store, and load physics data with minimal code  
✅ **AI-Ready Data** - Automatically formatted for neural network training  
✅ **Extensible** - Easy to add new physics systems  

## Quick Start

1. **Generate physics data:**
   ```python
   from src.simulate.oscillators import simulate_simple_harmonic
   simulate_simple_harmonic(k=1.0, m=1.0, x0=1.0, v0=0.0, n_steps=1000)
   ```

2. **Load data for AI:**
   ```python
   from src.dataloaders.database_physics_dataloader import PhysicsDataLoader
   loader = PhysicsDataLoader()
   states, derivatives = loader.get_training_data()
   ```

3. **Train your AI to discover equations from the data!**

## Complete Documentation

📖 **See [HOW_TO_USE.md](HOW_TO_USE.md) for detailed usage instructions and examples.**

## Project Structure

```
├── src/
│   ├── simulate/oscillators.py       # Physics simulators
│   ├── dataloaders/                  # Data loading utilities  
│   └── utils/enhanced_database.py    # Database management
├── db/simulations.sqlite             # All data stored here
├── generate_training_data.py         # Bulk data generation
├── test_simple_pipeline.py          # Test the pipeline
└── example_ai_training.py           # AI training example
```

## Ready for AI Development

The pipeline generates training data in the format:
- **Input**: `[position, velocity]` states
- **Output**: `[dx/dt, dv/dt]` derivatives  

Perfect for training neural networks to learn physics equations like:
- `dx/dt = velocity`
- `dv/dt = -(k/m) * position` (simple harmonic oscillator)

Start building your equation discovery AI! 🚀
