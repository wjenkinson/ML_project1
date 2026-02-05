from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main() -> None:
    """Plot vertical and horizontal centerline profiles of last-frame predictions vs ground truth.

    For each prediction sequence file saved by predict_sequence.py (pred_seq_{tag}.pt),
    this script:
    - Loads ground truth and predicted grids.
    - Extracts the vertical centerline from the last frame (column W//2).
    - Computes MSE between the full predicted grid and ground-truth grid.
    - Produces a single master figure overlaying all models' centerlines.
    """
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    seq_files = sorted(output_dir.glob("pred_seq_*.pt"))
    if not seq_files:
        print("No prediction sequence files found; run predict_sequence.py first.")
        return

    # Collect centerlines and MSEs for all models
    gt_vert_ref = None
    gt_horiz_ref = None
    y_idx_ref = None
    x_idx_ref = None
    model_curves = []  # list of (tag, vert_center_np, horiz_center_np, mse)

    for seq_path in seq_files:
        data = torch.load(seq_path, map_location="cpu")
        tag = data.get("tag")
        if not tag:
            stem = seq_path.stem  # e.g., 'pred_seq_cnn'
            tag = stem.replace("pred_seq_", "")

        gt_frames = data.get("gt")
        pred_frames = data.get("pred")

        if not gt_frames or not pred_frames:
            print(f"{seq_path} is missing 'gt' or 'pred' entries; skipping.")
            continue

        # Use the last available frame for this sequence
        gt_last = gt_frames[-1]  # (1, H, W)
        pred_last = pred_frames[-1]  # (1, H, W)

        if gt_last.shape != pred_last.shape:
            print(f"Shape mismatch for tag '{tag}': gt {gt_last.shape}, pred {pred_last.shape}; skipping.")
            continue

        # Compute MSE over the full grid
        mse = torch.mean((pred_last - gt_last) ** 2).item()

        # Extract vertical centerline (column W//2) and horizontal centerline (row H//2)
        _, H, W = gt_last.shape
        center_x = W // 2
        center_y = H // 2

        gt_vert = gt_last[0, :, center_x].detach().cpu().numpy()     # length H
        pred_vert = pred_last[0, :, center_x].detach().cpu().numpy()

        gt_horiz = gt_last[0, center_y, :].detach().cpu().numpy()    # length W
        pred_horiz = pred_last[0, center_y, :].detach().cpu().numpy()

        y_idx = list(range(H))
        x_idx = list(range(W))

        if gt_vert_ref is None:
            gt_vert_ref = gt_vert
            gt_horiz_ref = gt_horiz
            y_idx_ref = y_idx
            x_idx_ref = x_idx

        model_curves.append((tag, pred_vert, pred_horiz, mse))

    if gt_vert_ref is None or gt_horiz_ref is None or y_idx_ref is None or x_idx_ref is None or not model_curves:
        print("No valid centerline data collected; nothing to plot.")
        return

    # Create master figure with two subplots: left vertical, right horizontal
    fig, (ax_vert, ax_horiz) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Vertical centerline (y vs value)
    ax_vert.plot(y_idx_ref, gt_vert_ref, label="Ground truth", linewidth=2, color="black")
    for tag, pred_vert, _pred_horiz, mse in model_curves:
        ax_vert.plot(
            y_idx_ref,
            pred_vert,
            label=f"{tag.upper()} (MSE={mse:.4f})",
            linewidth=1.0,
        )
    ax_vert.set_xlabel("Grid y index (vertical centerline)")
    ax_vert.set_ylabel("Normalized density")
    ax_vert.set_title("Vertical centerline (last frame)")
    ax_vert.grid(True, alpha=0.3)
    ax_vert.legend()

    # Horizontal centerline (x vs value)
    ax_horiz.plot(x_idx_ref, gt_horiz_ref, label="Ground truth", linewidth=2, color="black")
    for tag, _pred_vert, pred_horiz, mse in model_curves:
        ax_horiz.plot(
            x_idx_ref,
            pred_horiz,
            label=f"{tag.upper()} (MSE={mse:.4f})",
            linewidth=1.0,
        )
    ax_horiz.set_xlabel("Grid x index (horizontal centerline)")
    ax_horiz.set_ylabel("Normalized density")
    ax_horiz.set_title("Horizontal centerline (last frame)")
    ax_horiz.grid(True, alpha=0.3)
    ax_horiz.legend()

    out_path = output_dir / "model_sensitivity_master.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved master centerline sensitivity plot to {out_path}")


if __name__ == "__main__":
    main()
