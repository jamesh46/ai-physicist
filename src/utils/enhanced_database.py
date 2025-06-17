"""
Enhanced database utilities for storing and retrieving physics simulation data.
All data is stored directly in SQLite - no external files needed.
"""
import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

class PhysicsDatabase:
    def __init__(self, db_path: str = "db/simulations.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables - store all data directly in DB."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Systems table - metadata about physical systems
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS systems (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    equation TEXT,
                    state_dim INTEGER NOT NULL,
                    parameters TEXT,  -- JSON string of parameters
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Trajectories table - ALL simulation data stored directly
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id INTEGER REFERENCES systems(id),
                    initial_conditions TEXT NOT NULL,  -- JSON array [x0, v0, ...]
                    time_data TEXT NOT NULL,           -- JSON array of time points
                    state_data TEXT NOT NULL,          -- JSON array of state vectors [[x1,v1],[x2,v2],...]
                    parameters TEXT,                   -- JSON string of simulation params
                    noise_level REAL DEFAULT 0.0,
                    dt REAL NOT NULL,
                    n_steps INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
    
    def add_system(self, name: str, equation: str, state_dim: int, 
                   parameters: Dict[str, Any] = None, description: str = None) -> int:
        """Add a new physical system to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO systems (name, equation, state_dim, parameters, description)
                VALUES (?, ?, ?, ?, ?)
            """, (name, equation, state_dim, 
                  json.dumps(parameters) if parameters else None, description))
            
            # Get the system ID
            cursor.execute("SELECT id FROM systems WHERE name = ?", (name,))
            return cursor.fetchone()[0]
    
    def add_trajectory(self, system_id: int, time_data: np.ndarray, 
                      state_data: np.ndarray, initial_conditions: np.ndarray,
                      dt: float, noise_level: float = 0.0, 
                      parameters: Dict[str, Any] = None) -> int:
        """Add a trajectory to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trajectories 
                (system_id, initial_conditions, time_data, state_data, 
                 parameters, noise_level, dt, n_steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                system_id,
                json.dumps(initial_conditions.tolist()),
                json.dumps(time_data.tolist()),
                json.dumps(state_data.tolist()),
                json.dumps(parameters) if parameters else None,
                noise_level,
                dt,
                len(time_data)
            ))
            return cursor.lastrowid
    
    def get_trajectory(self, trajectory_id: int) -> Optional[Dict[str, Any]]:
        """Get trajectory data by ID - returns numpy arrays."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trajectories WHERE id = ?", (trajectory_id,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                traj_data = dict(zip(columns, row))
                
                # Convert JSON strings back to numpy arrays
                traj_data['initial_conditions'] = np.array(json.loads(traj_data['initial_conditions']))
                traj_data['time_data'] = np.array(json.loads(traj_data['time_data']))
                traj_data['state_data'] = np.array(json.loads(traj_data['state_data']))
                if traj_data['parameters']:
                    traj_data['parameters'] = json.loads(traj_data['parameters'])
                
                return traj_data
        return None
    
    def get_system_trajectories(self, system_id: int) -> List[Dict[str, Any]]:
        """Get all trajectories for a given system."""
        trajectories = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM trajectories WHERE system_id = ?", (system_id,))
            for (traj_id,) in cursor.fetchall():
                trajectories.append(self.get_trajectory(traj_id))
        return trajectories
    
    def get_training_data(self, system_id: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all data formatted for AI training.
        Returns: (X, dX_dt) where X is states and dX_dt is derivatives.
        """
        all_states = []
        all_derivatives = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if system_id:
                cursor.execute("SELECT id FROM trajectories WHERE system_id = ?", (system_id,))
            else:
                cursor.execute("SELECT id FROM trajectories")
            
            trajectory_ids = [row[0] for row in cursor.fetchall()]
        
        for traj_id in trajectory_ids:
            traj = self.get_trajectory(traj_id)
            if traj:
                time_data = traj['time_data']
                state_data = traj['state_data']
                dt = traj['dt']
                
                # Calculate derivatives using finite differences
                derivatives = np.gradient(state_data, dt, axis=0)
                
                all_states.extend(state_data)
                all_derivatives.extend(derivatives)
        
        return np.array(all_states), np.array(all_derivatives)
    
    def list_systems(self) -> List[Dict[str, Any]]:
        """List all systems in the database."""
        systems = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, equation, state_dim, description FROM systems")
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                systems.append(dict(zip(columns, row)))
        return systems
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM systems")
            n_systems = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM trajectories")
            n_trajectories = cursor.fetchone()[0]
            cursor.execute("SELECT SUM(n_steps) FROM trajectories")
            total_points = cursor.fetchone()[0] or 0
            
        return {
            "systems": n_systems, 
            "trajectories": n_trajectories,
            "total_data_points": total_points
        }