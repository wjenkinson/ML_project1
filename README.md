# Frame Prediction with LAMMPS Simulation Data

This project focuses on predicting the next frame in a molecular dynamics simulation using machine learning. The goal is to learn the dynamics from LAMMPS simulation data.

## Project Structure

```
ML_project1/
├── data/                   # LAMMPS dump files
├── src/                    # Source code
│   └── explore_data.py     # Data exploration and visualization
├── output/                 # Output files and visualizations
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

1. First, explore the data:
   ```bash
   python src/explore_data.py
   ```
   This will:
   - Scan the data directory for LAMMPS dump files
   - Display basic information about the first and last frames
   - Generate visualizations in the `output` directory

## Next Steps

1. **Inspect the visualizations** in the `output` directory to understand the data
2. **Modify `explore_data.py`** to explore different aspects of the data
3. **Create a data loader** to prepare the data for training
4. **Implement a simple model** for frame prediction

## Data Format

The data consists of LAMMPS dump files in the `data` directory. Each file represents a snapshot of the simulation at a specific timestep.

## License

This project is for educational purposes.
