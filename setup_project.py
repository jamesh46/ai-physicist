#!/usr/bin/env python3
"""
Set up initial folders *and* create an SQLite DB with two starter tables:
  - systems   : one row per physical system (metadata)
  - trajectories : every simulated run, linked to its system
Run once after cloning the repo.
"""
import pathlib
import sqlite3

ROOT = pathlib.Path(__file__).resolve().parent
DB_PATH = ROOT / "db" / "simulations.sqlite"

FOLDERS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "notebooks",
    ROOT / "src" / "simulate",
    ROOT / "src" / "dataloaders",
    ROOT / "src" / "models",
    ROOT / "src" / "utils",
    ROOT / "tests",
    ROOT / "configs",
]

def create_dirs():
    for d in FOLDERS:
        d.mkdir(parents=True, exist_ok=True)
    print("✓ Folders verified/created")

def init_sqlite():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS systems (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            eqn         TEXT,             -- LaTeX or SymPy string
            state_dim   INTEGER,
            notes       TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trajectories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            system_id   INTEGER REFERENCES systems(id),
            t0          REAL,             -- start time
            dt          REAL,             -- timestep size
            n_steps     INTEGER,
            noise       REAL,             -- σ of noise added
            data_path   TEXT NOT NULL     -- relative path to .npz / .csv
        );
        """
    )
    con.commit()
    con.close()
    print(f"✓ SQLite DB created at {DB_PATH}")

if __name__ == "__main__":
    create_dirs()
    init_sqlite()
