from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LAMMPSDumpReader:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.timestep: int | None = None
        self.natoms: int = 0
        self.box_bounds: np.ndarray | None = None
        self.atoms: np.ndarray | None = None

    def read(self) -> np.ndarray:
        """Read a LAMMPS dump file and return atom data as a NumPy array."""
        with self.filepath.open("r", encoding="utf-8") as f:
            # Skip header lines
            for _ in range(3):
                f.readline()

            # Read timestep
            self.timestep = int(f.readline().strip())

            # Read number of atoms
            f.readline()  # Skip ITEM: NUMBER OF ATOMS
            self.natoms = int(f.readline().strip())

            # Read box bounds
            f.readline()  # Skip ITEM: BOX BOUNDS
            self.box_bounds = np.zeros((3, 2), dtype=np.float64)
            for i in range(3):
                self.box_bounds[i] = list(map(float, f.readline().strip().split()))

            # Read atom data
            f.readline()  # Skip ITEM: ATOMS ...
            data: List[List[float]] = []
            for _ in range(self.natoms):
                line = f.readline().strip().split()
                data.append([float(x) for x in line])

        self.atoms = np.asarray(data, dtype=np.float64)
        return self.atoms


def atoms_to_grid(
    atoms: np.ndarray,
    box_bounds: np.ndarray,
    grid_size: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """Rasterize atom coordinates into a 2D grid.

    Uses x and y coordinates to build a simple density map:
    - Each atom contributes +1 to a grid cell.
    - The grid is normalized to [0, 1] by dividing by the maximum cell count.
    """
    if atoms.ndim != 2 or atoms.shape[1] < 4:
        raise ValueError(f"Expected atom array of shape (N, >=4), got {atoms.shape}")

    if box_bounds.shape != (3, 2):
        raise ValueError(f"Expected box_bounds shape (3, 2), got {box_bounds.shape}")

    height, width = grid_size

    x_min, x_max = box_bounds[0]
    y_min, y_max = box_bounds[1]

    xs = atoms[:, 2]
    ys = atoms[:, 3]

    # Normalize to [0, 1]
    eps = 1e-9
    x_norm = (xs - x_min) / (x_max - x_min + eps)
    y_norm = (ys - y_min) / (y_max - y_min + eps)

    # Map to integer grid indices
    x_idx = np.clip((x_norm * (width - 1)).astype(int), 0, width - 1)
    y_idx = np.clip((y_norm * (height - 1)).astype(int), 0, height - 1)

    grid = np.zeros((height, width), dtype=np.float32)
    for xi, yi in zip(x_idx, y_idx):
        grid[yi, xi] += 1.0

    max_val = grid.max()
    if max_val > 0:
        grid /= max_val

    # Shape: (1, H, W) for a single-channel image-like tensor
    return torch.from_numpy(grid).unsqueeze(0)


class LammpsGridDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset yielding (frame_t, frame_t+1) grids for next-frame prediction."""

    def __init__(
        self,
        split: str,
        grid_size: Tuple[int, int] = (64, 64),
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.grid_size = grid_size

        project_root = Path(__file__).parent.parent
        self.data_dir = project_root / "data"
        splits_dir = self.data_dir / "splits"
        list_path = splits_dir / f"{split}_files.txt"

        if not list_path.exists():
            raise FileNotFoundError(f"Split file not found: {list_path}")

        with list_path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]

        def timestep_from_name(name: str) -> int:
            """Extract numeric timestep from a filename like 'dump.12345.LAMMPS'."""
            try:
                parts = name.split(".")
                return int(parts[1])
            except (IndexError, ValueError):
                return 0

        names_sorted = sorted(names, key=timestep_from_name)

        # Build (t, t+1) frame pairs
        self.frame_pairs: List[Tuple[str, str]] = []
        for i in range(len(names_sorted) - 1):
            self.frame_pairs.append((names_sorted[i], names_sorted[i + 1]))

    def __len__(self) -> int:
        return len(self.frame_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        in_name, out_name = self.frame_pairs[idx]

        in_reader = LAMMPSDumpReader(self.data_dir / in_name)
        in_atoms = in_reader.read()
        if in_reader.box_bounds is None:
            raise RuntimeError("box_bounds not set after reading input frame")
        grid_in = atoms_to_grid(in_atoms, in_reader.box_bounds, self.grid_size)

        out_reader = LAMMPSDumpReader(self.data_dir / out_name)
        out_atoms = out_reader.read()
        if out_reader.box_bounds is None:
            raise RuntimeError("box_bounds not set after reading target frame")
        grid_out = atoms_to_grid(out_atoms, out_reader.box_bounds, self.grid_size)

        return grid_in, grid_out


def main() -> None:
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    train_dataset = LammpsGridDataset(split="train", grid_size=(64, 64))
    print(f"Train dataset length (pairs): {len(train_dataset)}")

    sample_in, sample_out = train_dataset[0]
    print("Sample input tensor shape:", sample_in.shape)
    print("Sample target tensor shape:", sample_out.shape)

    val_dataset = LammpsGridDataset(split="val", grid_size=(64, 64))
    print(f"Validation dataset length (pairs): {len(val_dataset)}")


if __name__ == "__main__":
    main()
