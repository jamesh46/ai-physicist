# How to Use the AI Physicist Project

This guide shows you how to use the streamlined physics simulation and AI training pipeline. The entire project uses only an SQLite database - no external files needed.

## Quick Start

### 1. Set Up the Environment
```bash
# Install required packages
pip install numpy scipy

# The database will be created automatically when you first run a script
```

### 2. Generate Physics Data
```python
# Generate some simple harmonic oscillator data
from src.simulate.oscillators import simulate_simple_harmonic, simulate_damped_oscillator

# Create a few trajectories with different parameters
simulate_simple_harmonic(k=1.0, m=1.0, x0=1.0, v0=0.0, n_steps=1000)
simulate_simple_harmonic(k=2.0, m=1.0, x0=0.5, v0=0.5, n_steps=1000)
simulate_damped_oscillator(omega=1.0, zeta=0.1, x0=1.0, v0=0.0, n_steps=1000)
```

Or use the bulk data generation script:
```bash
python generate_training_data.py
```

### 3. Load Data for AI Training
```python
from src.dataloaders.database_physics_dataloader import PhysicsDataLoader

# Load all physics data
loader = PhysicsDataLoader()
states, derivatives = loader.get_training_data()

# states: (N, 2) array of [position, velocity] 
# derivatives: (N, 2) array of [dx/dt, dv/dt]
```

## Core Components

## Core Components

The project has three main modules:

### 🗄️ Database (`src/utils/enhanced_database.py`)
- **PhysicsDatabase**: Stores all simulation data directly in SQLite
- No external files - everything in one database
- Automatic derivative calculation for AI training

### 🔬 Physics Simulators (`src/simulate/oscillators.py`)
- **`simulate_simple_harmonic(k, m, x0, v0, ...)`**: Creates SHO data
- **`simulate_damped_oscillator(omega, zeta, x0, v0, ...)`**: Creates damped oscillator data
- Returns trajectory ID for database reference

### 📊 Data Loader (`src/dataloaders/database_physics_dataloader.py`)
- **PhysicsDataLoader**: Simple interface for AI training data
- **`get_training_data()`**: Returns (states, derivatives) arrays
- **`get_system_data(system_name)`**: Get data for specific physics systems

## Common Usage Patterns

### Generate Diverse Training Data
```python
import numpy as np
from src.simulate.oscillators import simulate_simple_harmonic, simulate_damped_oscillator

# Generate multiple trajectories with random parameters
for i in range(20):
    k = np.random.uniform(0.5, 3.0)
    m = np.random.uniform(0.5, 2.0) 
    x0 = np.random.uniform(-2.0, 2.0)
    v0 = np.random.uniform(-1.0, 1.0)
    noise = np.random.uniform(0.0, 0.05)
    
    simulate_simple_harmonic(k=k, m=m, x0=x0, v0=v0, n_steps=1000, noise=noise)
```

### Prepare Data for Neural Networks
```python
from src.dataloaders.database_physics_dataloader import PhysicsDataLoader
import numpy as np

loader = PhysicsDataLoader()

# Get training data
states, derivatives = loader.get_training_data()

# Normalize for neural network training
states_mean = states.mean(axis=0)
states_std = states.std(axis=0)
states_normalized = (states - states_mean) / states_std

derivatives_mean = derivatives.mean(axis=0)
derivatives_std = derivatives.std(axis=0)
derivatives_normalized = (derivatives - derivatives_mean) / derivatives_std

# Split into train/validation
n_samples = len(states)
train_size = int(0.8 * n_samples)
indices = np.random.permutation(n_samples)

X_train = states_normalized[indices[:train_size]]
y_train = derivatives_normalized[indices[:train_size]]
X_val = states_normalized[indices[train_size:]]
y_val = derivatives_normalized[indices[train_size:]]
```

### Inspect Your Data
```python
from src.dataloaders.database_physics_dataloader import PhysicsDataLoader

loader = PhysicsDataLoader()

# Print summary
loader.print_summary()

# Get specific system data
sho_data = loader.get_system_data("simple_harmonic")
damped_data = loader.get_system_data("damped_oscillator")

# Check trajectory details
trajectory = loader.get_trajectory_data(trajectory_id=1)
print(f"Initial conditions: {trajectory['initial_conditions']}")
print(f"Time range: {trajectory['time_data'][0]} to {trajectory['time_data'][-1]}")
```

## Database Schema
```sql
-- Systems table: metadata about physical systems
CREATE TABLE systems (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    equation TEXT,
    state_dim INTEGER NOT NULL,
    parameters TEXT,  -- JSON string
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trajectories table: ALL simulation data stored directly
CREATE TABLE trajectories (
    id INTEGER PRIMARY KEY,
    system_id INTEGER REFERENCES systems(id),
    initial_conditions TEXT NOT NULL,  -- JSON array [x0, v0, ...]
    time_data TEXT NOT NULL,           -- JSON array of time points
    state_data TEXT NOT NULL,          -- JSON array [[x1,v1],[x2,v2],...]
    parameters TEXT,                   -- JSON simulation params
    noise_level REAL DEFAULT 0.0,
    dt REAL NOT NULL,
    n_steps INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Available Scripts

### 🚀 **`test_simple_pipeline.py`**
Test the basic pipeline - generates a few trajectories and loads them back.
```bash
python test_simple_pipeline.py
```

### 📈 **`generate_training_data.py`** 
Generate a diverse dataset with 60 trajectories (30 SHO + 30 damped).
```bash
python generate_training_data.py
```

### 🤖 **`example_ai_training.py`**
Shows how to load and preprocess data for AI model training.
```bash
python example_ai_training.py
```

## Database Schema Reference

The SQLite database has two tables:

```sql
-- Systems: Physics system metadata
CREATE TABLE systems (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,           -- "simple_harmonic", "damped_oscillator"
    equation TEXT,                       -- LaTeX/text equation
    state_dim INTEGER NOT NULL,          -- 2 for [position, velocity]
    parameters TEXT,                     -- JSON: {"k": 1.0, "m": 1.0}
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trajectories: All simulation data stored as JSON
CREATE TABLE trajectories (
    id INTEGER PRIMARY KEY,
    system_id INTEGER REFERENCES systems(id),
    initial_conditions TEXT NOT NULL,    -- JSON: [x0, v0]
    time_data TEXT NOT NULL,            -- JSON: [0.0, 0.01, 0.02, ...]
    state_data TEXT NOT NULL,           -- JSON: [[x1,v1], [x2,v2], ...]
    parameters TEXT,                    -- JSON: simulation parameters
    noise_level REAL DEFAULT 0.0,
    dt REAL NOT NULL,                   -- timestep
    n_steps INTEGER NOT NULL,           -- number of time points
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Building AI Models

### Step 1: Neural Network Architecture
Your AI should learn the function: `f(position, velocity) → (dx/dt, dv/dt)`

```python
# Example with PyTorch/TensorFlow
# Input: [position, velocity] (2D)
# Output: [dx/dt, dv/dt] (2D)

# The network should discover:
# dx/dt = velocity
# dv/dt = -k/m * position  (for simple harmonic oscillator)
```

### Step 2: Training Data Format
```python
# X: states of shape (N, 2) where columns are [position, velocity]
# y: derivatives of shape (N, 2) where columns are [dx/dt, dv/dt]
states, derivatives = loader.get_training_data()
```

### Step 3: Equation Discovery
After training your neural network:
1. Use symbolic regression to extract equations from the learned function
2. Compare discovered equations with known physics: `x'' = -(k/m)x`
3. Validate on held-out test trajectories

## Tips for Success

✅ **Start Simple**: Use the test script first to understand the data flow  
✅ **Normalize Data**: Neural networks train better on normalized inputs  
✅ **Use Derivatives**: The derivatives are automatically calculated for you  
✅ **Check Data Quality**: Use `loader.print_summary()` to inspect your dataset  
✅ **Add More Physics**: Extend with pendulums, springs, etc. by following the oscillator examples  

## Troubleshooting

**Q: No data found?**  
A: Run `python generate_training_data.py` first to create the dataset.

**Q: Import errors?**  
A: Make sure you're running from the project root directory.

**Q: Database file missing?**  
A: The database is created automatically when you first run a simulation.

---

🎯 **Ready to build your AI physicist!** The data pipeline is complete - now focus on the neural networks and equation discovery algorithms.

## Final Project Structure

The project is now streamlined to just the essential files:

```
ai-physicist/
├── src/
│   ├── utils/enhanced_database.py           # Database management
│   ├── simulate/oscillators.py              # Physics simulators  
│   └── dataloaders/database_physics_dataloader.py  # Data loading
├── db/simulations.sqlite                    # ALL data stored here
├── test_simple_pipeline.py                 # Test the pipeline
├── generate_training_data.py               # Bulk data generation
├── example_ai_training.py                  # AI training example
├── HOW_TO_USE.md                          # This guide
├── README.md                              # Quick overview
└── requirements.txt                       # Dependencies
```

**Total: 8 essential files + 1 database file = Everything you need!** 🚀
