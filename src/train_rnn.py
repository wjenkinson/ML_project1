from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from grid_dataset import LAMMPSDumpReader, atoms_to_grid


class RnnSequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Sequence dataset for RNNs over rasterized grids.

    For a given split (train/val), it:
    - Loads all frames listed in data/splits/{split}_files.txt
    - Converts each frame to a (1, H, W) grid
    - Builds sliding windows of length seq_len as inputs and the
      next frame as target.

    Each sample is:
    - input_seq: (seq_len, input_size) flattened grids
    - target: (input_size,) flattened next grid
    """

    def __init__(self, split: str, seq_len: int = 4, grid_size: Tuple[int, int] = (64, 64)) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.seq_len = seq_len
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
            try:
                parts = name.split(".")
                return int(parts[1])
            except (IndexError, ValueError):
                return 0

        names_sorted = sorted(names, key=timestep_from_name)

        # Precompute all grids in memory for this split
        self.frames: List[torch.Tensor] = []
        for name in names_sorted:
            reader = LAMMPSDumpReader(self.data_dir / name)
            atoms = reader.read()
            if reader.box_bounds is None:
                raise RuntimeError("box_bounds not set after reading frame")
            grid = atoms_to_grid(atoms, reader.box_bounds, self.grid_size)
            self.frames.append(grid)  # (1, H, W)

        self.input_size = self.frames[0].numel() if self.frames else 0

    def __len__(self) -> int:
        # Number of sequences of length seq_len with a next-frame target
        return max(0, len(self.frames) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build sequence of length seq_len and next-frame target
        seq_frames = self.frames[idx : idx + self.seq_len]        # list of (1, H, W)
        target_frame = self.frames[idx + self.seq_len]            # (1, H, W)

        # Stack to (seq_len, 1, H, W)
        seq = torch.stack(seq_frames, dim=0)

        # Flatten grids for RNN: (seq_len, input_size)
        seq_flat = seq.view(self.seq_len, -1)
        target_flat = target_frame.view(-1)

        return seq_flat, target_flat


class SimpleRnnPredictor(nn.Module):
    """Vanilla RNN that predicts the next flattened grid from a sequence."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        pred = self.fc(last_hidden)  # (batch, input_size)
        return pred


def train(
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seq_len: int = 4,
) -> None:
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(0)

    train_dataset = RnnSequenceDataset(split="train", seq_len=seq_len, grid_size=(64, 64))
    val_dataset = RnnSequenceDataset(split="val", seq_len=seq_len, grid_size=(64, 64))

    if train_dataset.input_size == 0:
        print("Train dataset is empty; nothing to train on.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleRnnPredictor(input_size=train_dataset.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "simple_rnn_predictor.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)   # (batch, seq_len, input_size)
            targets = targets.to(device) # (batch, input_size)

            optimizer.zero_grad(set_to_none=True)
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                preds = model(inputs)
                loss = criterion(preds, targets)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved new best vanilla RNN model to {model_path}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
