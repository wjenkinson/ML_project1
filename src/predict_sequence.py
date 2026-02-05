from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_dataset import LammpsGridDataset, atoms_to_grid
from graph_dataset import LammpsGraphDataset
from train_cnn import SimpleFramePredictor
from train_rnn import SimpleRnnPredictor
from train_gru import SimpleGruPredictor
from train_lstm import SimpleLstmPredictor
from train_gnn import SimpleGnnPredictor


def save_grid(path: Path, grid: torch.Tensor, title: str) -> None:
    """Save a single (1, H, W) grid as an image."""
    grid_np = grid.squeeze(0).detach().cpu().numpy()

    plt.figure(figsize=(4, 4))
    im = plt.imshow(grid_np, cmap="viridis", origin="lower")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def extract_model_tag(ckpt_path: Path) -> str:
    """Derive a short model tag from a checkpoint filename.

    For example, 'simple_cnn_predictor.pt' -> 'cnn'.
    """
    name = ckpt_path.stem  # e.g., 'simple_cnn_predictor'
    if name.startswith("simple_") and name.endswith("_predictor"):
        return name[len("simple_") : -len("_predictor")]
    return name


def predict_with_cnn(
    ckpt_path: Path,
    device: torch.device,
    gt_frames: List[torch.Tensor],
    num_steps: int,
) -> List[torch.Tensor]:
    """Autoregressively roll out predictions using the CNN model."""

    model = SimpleFramePredictor().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if not gt_frames:
        return []

    current = gt_frames[0].unsqueeze(0).to(device)  # (1, 1, H, W)
    pred_frames: List[torch.Tensor] = []

    for _ in range(num_steps):
        with torch.no_grad():
            pred = model(current)  # (1, 1, H, W)

        pred_frames.append(pred.squeeze(0).detach().cpu())  # (1, H, W)
        current = pred

    return pred_frames


def predict_with_gnn(
    ckpt_path: Path,
    device: torch.device,
    gt_frames: List[torch.Tensor],
    num_steps: int,
) -> List[torch.Tensor]:
    """Predict next-frame grids using the GNN model on the validation split.

    This uses the LammpsGraphDataset('val') to obtain particle positions at time t,
    applies the trained GNN to predict positions at t+1, and then rasterizes the
    predicted positions back into 2D grids via atoms_to_grid.

    Note: This function performs one-step predictions per validation pair (t, t+1),
    not an autoregressive rollout. The returned list length is len(val_graph_dataset).
    """

    # Load validation graph dataset (uses all particles, radius-based edges)
    val_graph = LammpsGraphDataset(split="val")
    if len(val_graph) == 0:
        print("  Validation graph dataset is empty; nothing to predict for GNN.")
        return []

    # Instantiate and load the trained GNN
    model = SimpleGnnPredictor(in_channels=4, hidden_channels=64, num_layers=2).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    pred_frames: List[torch.Tensor] = []

    for idx in range(len(val_graph)):
        data = val_graph[idx]
        data = data.to(device)

        with torch.no_grad():
            pred_pos = model(data)  # (N, 3)

        pred_pos_cpu = pred_pos.detach().cpu()
        atom_type = data.atom_type.cpu().numpy()
        box_bounds = data.box_bounds.cpu().numpy()

        # Build a pseudo-"atoms" array compatible with atoms_to_grid:
        # columns: [id, type, x, y, z, ...]
        pos_np = pred_pos_cpu.numpy()
        n_atoms = pos_np.shape[0]
        atoms_arr = np.zeros((n_atoms, 6), dtype=np.float64)
        atoms_arr[:, 0] = np.arange(1, n_atoms + 1, dtype=np.float64)  # dummy IDs
        atoms_arr[:, 1] = atom_type.astype(np.float64)
        atoms_arr[:, 2:5] = pos_np[:, :3]

        grid = atoms_to_grid(atoms_arr, box_bounds, grid_size=(64, 64))  # (1, H, W)
        pred_frames.append(grid)

    return pred_frames


def predict_with_gru(
    ckpt_path: Path,
    device: torch.device,
    gt_frames: List[torch.Tensor],
    num_steps: int,
    seq_len: int = 4,
) -> List[torch.Tensor]:

    if len(gt_frames) <= seq_len:
        print("  Not enough frames for GRU rollout; need more than seq_len.")
        return []

    input_size = gt_frames[0].numel()

    model = SimpleGruPredictor(input_size=input_size).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    seq_flat = torch.stack(
        [f.view(-1) for f in gt_frames[:seq_len]], dim=0
    )

    pred_frames: List[torch.Tensor] = []

    for _ in range(num_steps):
        inp = seq_flat.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_flat = model(inp)[0]

        pred_grid = pred_flat.view_as(gt_frames[0])
        pred_frames.append(pred_grid.detach().cpu())

        seq_flat = torch.cat([seq_flat[1:], pred_flat.unsqueeze(0).cpu()], dim=0)

    return pred_frames


def predict_with_lstm(
    ckpt_path: Path,
    device: torch.device,
    gt_frames: List[torch.Tensor],
    num_steps: int,
    seq_len: int = 4,
) -> List[torch.Tensor]:

    if len(gt_frames) <= seq_len:
        print("  Not enough frames for LSTM rollout; need more than seq_len.")
        return []

    input_size = gt_frames[0].numel()

    model = SimpleLstmPredictor(input_size=input_size).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    seq_flat = torch.stack(
        [f.view(-1) for f in gt_frames[:seq_len]], dim=0
    )

    pred_frames: List[torch.Tensor] = []

    for _ in range(num_steps):
        inp = seq_flat.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_flat = model(inp)[0]

        pred_grid = pred_flat.view_as(gt_frames[0])
        pred_frames.append(pred_grid.detach().cpu())

        seq_flat = torch.cat([seq_flat[1:], pred_flat.unsqueeze(0).cpu()], dim=0)

    return pred_frames


def predict_with_rnn(
    ckpt_path: Path,
    device: torch.device,
    gt_frames: List[torch.Tensor],
    num_steps: int,
    seq_len: int = 4,
) -> List[torch.Tensor]:
    """Autoregressively roll out predictions using the vanilla RNN model.

    The RNN was trained on sequences of length ``seq_len`` of flattened grids
    to predict the next flattened grid. Here we:
    - start from the first ``seq_len`` ground-truth frames,
    - repeatedly predict the next frame, and
    - slide the input window forward using the prediction.
    """

    if len(gt_frames) <= seq_len:
        print("  Not enough frames for RNN rollout; need more than seq_len.")
        return []

    # All grids have the same shape; flatten to get input_size
    input_size = gt_frames[0].numel()

    model = SimpleRnnPredictor(input_size=input_size).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Initial sequence: first seq_len ground-truth frames
    seq_flat = torch.stack(
        [f.view(-1) for f in gt_frames[:seq_len]], dim=0
    )  # (seq_len, input_size)

    pred_frames: List[torch.Tensor] = []

    for _ in range(num_steps):
        inp = seq_flat.unsqueeze(0).to(device)  # (1, seq_len, input_size)
        with torch.no_grad():
            pred_flat = model(inp)[0]  # (input_size,)

        pred_grid = pred_flat.view_as(gt_frames[0])  # (1, H, W)
        pred_frames.append(pred_grid.detach().cpu())

        # Slide window: drop oldest, append new prediction
        seq_flat = torch.cat([seq_flat[1:], pred_flat.unsqueeze(0).cpu()], dim=0)

    return pred_frames


def main() -> None:
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cpu")

    # Shared validation dataset and ground-truth frames
    val_dataset = LammpsGridDataset(split="val", grid_size=(64, 64))
    if len(val_dataset) == 0:
        print("Validation dataset is empty; nothing to predict.")
        return

    # Build ground-truth frames: t0 input, then all targets
    start_in, _ = val_dataset[0]
    gt_frames: List[torch.Tensor] = [start_in]
    for i in range(len(val_dataset)):
        _, gt = val_dataset[i]
        gt_frames.append(gt)

    num_steps = 5

    # Discover available model checkpoints
    ckpt_files = sorted(output_dir.glob("simple_*_predictor.pt"))
    if not ckpt_files:
        print(f"No model checkpoints matching 'simple_*_predictor.pt' found in {output_dir}")
        return

    for ckpt in ckpt_files:
        tag = extract_model_tag(ckpt)
        print(f"\nRunning prediction sequence for model tag='{tag}' from {ckpt.name}")

        if tag == "cnn":
            pred_frames = predict_with_cnn(ckpt, device, gt_frames, num_steps)
        elif tag == "rnn":
            pred_frames = predict_with_rnn(ckpt, device, gt_frames, num_steps)
        elif tag == "gru":
            pred_frames = predict_with_gru(ckpt, device, gt_frames, num_steps)
        elif tag == "lstm":
            pred_frames = predict_with_lstm(ckpt, device, gt_frames, num_steps)
        elif tag == "gnn":
            pred_frames = predict_with_gnn(ckpt, device, gt_frames, num_steps)
        else:
            print(f"  Skipping unsupported model tag '{tag}'")
            continue

        if not pred_frames:
            print("  No predictions generated; skipping.")
            continue

        # Align ground truth and predictions to the same length
        seq_len = min(len(pred_frames), len(gt_frames) - 1)
        gt_seq = gt_frames[1 : seq_len + 1]
        pred_seq = pred_frames[:seq_len]

        seq_path = output_dir / f"pred_seq_{tag}.pt"
        torch.save(
            {
                "tag": tag,
                "gt": [f.cpu() for f in gt_seq],
                "pred": [f.cpu() for f in pred_seq],
            },
            seq_path,
        )
        print(f"  Saved prediction sequence to {seq_path}")

    print("\nFinished generating prediction sequences.")


if __name__ == "__main__":
    main()
