import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class LAMMPSDumpReader:
    """Simple LAMMPS dump file reader for visualization."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.timestep = None
        self.natoms = 0
        self.box_bounds = None
        self.atoms = None
        
    def read(self):
        """Read a LAMMPS dump file and return atom data."""
        with open(self.filepath, 'r') as f:
            # Skip header lines
            for _ in range(3):  # Skip first 3 lines (ITEM: TIMESTEP, ITEM: NUMBER OF ATOMS, ITEM: BOX BOUNDS)
                f.readline()
                
            # Read timestep
            self.timestep = int(f.readline().strip())
            
            # Read number of atoms
            f.readline()  # Skip ITEM: NUMBER OF ATOMS
            self.natoms = int(f.readline().strip())
            
            # Read box bounds
            f.readline()  # Skip ITEM: BOX BOUNDS
            self.box_bounds = np.zeros((3, 2))
            for i in range(3):
                self.box_bounds[i] = list(map(float, f.readline().strip().split()))
            
            # Read atom data
            f.readline()  # Skip ITEM: ATOMS ...
            data = []
            for _ in range(self.natoms):
                line = f.readline().strip().split()
                data.append([float(x) for x in line])
                
            self.atoms = np.array(data)
            return self.atoms
    
    def visualize(self, save_path=None):
        """Create a simple 2D visualization of the atom positions."""
        if self.atoms is None:
            self.read()
            
        plt.figure(figsize=(10, 8))
        
        # Assuming the first three columns are atom ID, type, x, y, z
        if self.atoms.shape[1] >= 4:  # Make sure we have at least x, y, z coordinates
            plt.scatter(self.atoms[:, 2], self.atoms[:, 3], s=10, alpha=0.6)
            plt.title(f'LAMMPS Simulation Snapshot (Timestep: {self.timestep})')
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Insufficient data for 2D visualization. Need at least x, y coordinates.")

def main():
    # Get the data directory
    data_dir = Path(__file__).parent.parent / 'data'
    
    # List all LAMMPS dump files
    dump_files = sorted(list(data_dir.glob('dump.*.LAMMPS')))
    
    if not dump_files:
        print("No LAMMPS dump files found in the data directory.")
        return
    
    print(f"Found {len(dump_files)} LAMMPS dump files.")
    
    # Read and visualize the first and last frames
    for filepath in [dump_files[0], dump_files[-1]]:
        print(f"\nAnalyzing {filepath.name}")
        reader = LAMMPSDumpReader(filepath)
        atoms = reader.read()
        print(f"Timestep: {reader.timestep}")
        print(f"Number of atoms: {reader.natoms}")
        print(f"Box bounds:\n{reader.box_bounds}")
        print(f"Atom data shape: {atoms.shape}")
        
        # Create output directory for visualizations
        output_dir = Path(__file__).parent.parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # Save visualization
        vis_path = output_dir / f"visualization_{filepath.stem}.png"
        reader.visualize(save_path=str(vis_path))
        print(f"Visualization saved to {vis_path}")

if __name__ == "__main__":
    main()
