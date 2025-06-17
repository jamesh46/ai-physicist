"""
Simple data loader for physics simulation data from SQLite database.
"""
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Optional, List, Dict, Any

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from utils.enhanced_database import PhysicsDatabase


class PhysicsDataLoader:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = ROOT / "db" / "simulations.sqlite"
        self.db = PhysicsDatabase(str(db_path))
    
    def get_training_data(self, system_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data for AI models.
        Returns: (states, derivatives) for training equation discovery models.
        """
        if system_name:
            # Get specific system
            systems = self.db.list_systems()
            system_id = None
            for sys in systems:
                if sys['name'] == system_name:
                    system_id = sys['id']
                    break
            if system_id is None:
                raise ValueError(f"System '{system_name}' not found")
            return self.db.get_training_data(system_id)
        else:
            # Get all data
            return self.db.get_training_data()
    
    def get_trajectory_data(self, trajectory_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific trajectory by ID."""
        return self.db.get_trajectory(trajectory_id)
    
    def list_available_systems(self) -> List[Dict[str, Any]]:
        """List all available physics systems."""
        return self.db.list_systems()
    
    def get_system_data(self, system_name: str) -> List[Dict[str, Any]]:
        """Get all trajectories for a specific system."""
        systems = self.db.list_systems()
        system_id = None
        for sys in systems:
            if sys['name'] == system_name:
                system_id = sys['id']
                break
        
        if system_id is None:
            raise ValueError(f"System '{system_name}' not found")
            
        return self.db.get_system_trajectories(system_id)
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        return self.db.get_stats()
    
    def print_summary(self):
        """Print a summary of available data."""
        stats = self.get_stats()
        systems = self.list_available_systems()
        
        print(f"Database Summary:")
        print(f"- Total systems: {stats['systems']}")
        print(f"- Total trajectories: {stats['trajectories']}")
        print(f"- Total data points: {stats['total_data_points']}")
        print("\nAvailable systems:")
        for sys in systems:
            print(f"  - {sys['name']}: {sys['equation']}")