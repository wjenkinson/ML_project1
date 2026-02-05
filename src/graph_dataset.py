from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torch_geometric.data import Data

from grid_dataset import LAMMPSDumpReader


class LammpsGraphDataset(Dataset):
    """Graph dataset for particle-based simulations using all particles.

    For a given split (train/val), this dataset:
    - Reads the corresponding dump filenames from data/splits/{split}_files.txt.
    - Sorts them by numeric timestep.
    - Loads all frames into memory (positions, types, box bounds), sorted by atom ID
      so that nodes are consistently ordered across time.
    - Builds consecutive frame pairs (t, t+1).

    Each sample is a torch_geometric.data.Data object with fields:
    - x: node features [x, y, z, type_id] at time t, shape (N, 4)
    - pos: positions at time t, shape (N, 3)
    - edge_index: graph connectivity from radius-based neighborhood at time t
    - y: target positions at time t+1, shape (N, 3)
    - atom_type: integer atom type IDs, shape (N,)
    - box_bounds: simulation box bounds as (3, 2) tensor
    """

    def __init__(
        self,
        split: str,
        radius: float = 0.002,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.radius = radius

        project_root = Path(__file__).parent.parent
        self.data_dir = project_root / "data"
        splits_dir = self.data_dir / "splits"
        list_path = splits_dir / f"{split}_files.txt"

        if not list_path.exists():
            raise FileNotFoundError(f"Split file not found: {list_path}")

        with list_path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]

        def timestep_from_name(name: str) -> int:
            try:
                parts = name.split(".")
                return int(parts[1])
            except (IndexError, ValueError):
                return 0

        names_sorted = sorted(names, key=timestep_from_name)

        # Load all frames into memory, keyed by filename
        self.frames: Dict[str, Dict[str, torch.Tensor]] = {}

        for name in names_sorted:
            reader = LAMMPSDumpReader(self.data_dir / name)
            atoms = reader.read()
            if reader.box_bounds is None:
                raise RuntimeError(f"box_bounds not set after reading frame {name}")

            # atoms columns: [id, type, x, y, z, ...]
            if atoms.shape[1] < 5:
                raise ValueError(f"Expected at least 5 columns in atoms for {name}, got {atoms.shape[1]}")

            ids = atoms[:, 0].astype(int)
            sort_idx = np.argsort(ids)
            atoms_sorted = atoms[sort_idx]

            pos = torch.from_numpy(atoms_sorted[:, 2:5].astype(np.float32))  # (N, 3)
            atom_type = torch.from_numpy(atoms_sorted[:, 1].astype(np.int64))  # (N,)
            box_bounds = torch.from_numpy(reader.box_bounds.astype(np.float32))  # (3, 2)

            self.frames[name] = {
                "pos": pos,
                "atom_type": atom_type,
                "box_bounds": box_bounds,
            }

        # Build consecutive pairs (t, t+1)
        self.pairs: List[Tuple[str, str]] = []
        for i in range(len(names_sorted) - 1):
            self.pairs.append((names_sorted[i], names_sorted[i + 1]))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Data:  # type: ignore[override]
        name_t, name_tp1 = self.pairs[idx]

        frame_t = self.frames[name_t]
        frame_tp1 = self.frames[name_tp1]

        pos_t = frame_t["pos"]  # (N, 3)
        atom_type = frame_t["atom_type"]  # (N,)
        box_bounds = frame_t["box_bounds"]  # (3, 2)

        pos_tp1 = frame_tp1["pos"]  # (N, 3)

        if pos_t.shape != pos_tp1.shape:
            raise ValueError(
                f"Mismatched shapes between {name_t} and {name_tp1}: "
                f"{pos_t.shape} vs {pos_tp1.shape}"
            )

        # Node features: [x, y, z, type_id]
        x = torch.cat([pos_t, atom_type.to(dtype=torch.float32).unsqueeze(-1)], dim=-1)

        # Radius-based neighborhood graph (first neighbors) using scikit-learn
        # This avoids the need for torch-cluster.
        coords = pos_t.cpu().numpy()  # (N, 3)
        nbrs = NearestNeighbors(radius=self.radius, algorithm="ball_tree")
        nbrs.fit(coords)
        # indices[i] is an array of neighbor indices for node i (including itself)
        indices = nbrs.radius_neighbors(return_distance=False)

        row: List[int] = []
        col: List[int] = []
        for i, neigh in enumerate(indices):
            for j in neigh:
                if i == j:
                    continue  # no self-loops (loop=False)
                row.append(i)
                col.append(j)

        edge_index = torch.tensor([row, col], dtype=torch.long)

        data = Data(
            x=x,
            pos=pos_t,
            edge_index=edge_index,
            y=pos_tp1,
            atom_type=atom_type,
            box_bounds=box_bounds,
        )

        return data
