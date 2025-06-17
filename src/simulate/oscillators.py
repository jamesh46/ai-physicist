"""
Simplified physics simulation that generates data directly to the database.
No external files - everything stored in SQLite.
"""
import numpy as np
from scipy.integrate import odeint
from pathlib import Path
import sys

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from utils.enhanced_database import PhysicsDatabase


def simulate_simple_harmonic(
    k: float = 1.0,
    m: float = 1.0,
    x0: float = 1.0,
    v0: float = 0.0,
    t0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    noise: float = 0.0
) -> int:
    """
    Simulate a simple harmonic oscillator: x'' = - (k/m) * x
    Returns trajectory ID from database.
    """
    db = PhysicsDatabase()
    
    # Add system if not exists
    system_id = db.add_system(
        name="simple_harmonic",
        equation="x'' = - (k/m) * x",
        state_dim=2,
        parameters={"k": k, "m": m},
        description="Simple harmonic oscillator"
    )
    
    # time vector
    t = t0 + np.arange(n_steps) * dt

    # ODE definition
    def sho(state, t):
        x, v = state
        return [v, - (k/m) * x]

    # integrate
    sol = odeint(sho, [x0, v0], t)
    state_data = sol  # [position, velocity] for each time step

    # add noise
    if noise > 0:
        state_data += np.random.normal(scale=noise, size=state_data.shape)

    # Store in database
    trajectory_id = db.add_trajectory(
        system_id=system_id,
        time_data=t,
        state_data=state_data,
        initial_conditions=np.array([x0, v0]),
        dt=dt,
        noise_level=noise,
        parameters={"k": k, "m": m}
    )
    
    print(f"✓ SHO saved to database: trajectory_id={trajectory_id}")
    return trajectory_id


def simulate_damped_oscillator(
    omega: float = 1.0,
    zeta: float = 0.1,
    x0: float = 1.0,
    v0: float = 0.0,
    t0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    noise: float = 0.0
) -> int:
    """
    Simulate a damped harmonic oscillator: x'' + 2*zeta*omega*x' + omega^2*x = 0
    Returns trajectory ID from database.
    """
    db = PhysicsDatabase()
    
    # Add system if not exists
    system_id = db.add_system(
        name="damped_oscillator",
        equation="x'' + 2*zeta*omega*x' + omega^2*x = 0",
        state_dim=2,
        parameters={"omega": omega, "zeta": zeta},
        description="Damped harmonic oscillator"
    )
    
    # time vector
    t = t0 + np.arange(n_steps) * dt

    # ODE definition
    def damped(state, t):
        x, v = state
        return [v, -2*zeta*omega*v - (omega**2)*x]

    # integrate
    sol = odeint(damped, [x0, v0], t)
    state_data = sol  # [position, velocity] for each time step

    # add noise
    if noise > 0:
        state_data += np.random.normal(scale=noise, size=state_data.shape)

    # Store in database
    trajectory_id = db.add_trajectory(
        system_id=system_id,
        time_data=t,
        state_data=state_data,
        initial_conditions=np.array([x0, v0]),
        dt=dt,
        noise_level=noise,
        parameters={"omega": omega, "zeta": zeta}
    )
    
    print(f"✓ Damped oscillator saved to database: trajectory_id={trajectory_id}")
    return trajectory_id
