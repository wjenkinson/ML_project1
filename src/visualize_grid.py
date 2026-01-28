from pathlib import Path

import matplotlib.pyplot as plt
import torch

from grid_dataset import LammpsGridDataset


def main() -> None:
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    dataset = LammpsGridDataset(split="train", grid_size=(64, 64))
    if len(dataset) == 0:
        print("Train dataset is empty; nothing to visualize.")
        return

    grid_in, grid_out = dataset[0]

    # grid_in and grid_out are tensors with shape (1, H, W)
    grid_in_np = grid_in.squeeze(0).detach().cpu().numpy()
    grid_out_np = grid_out.squeeze(0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(grid_in_np, cmap="viridis", origin="lower")
    axes[0].set_title("Input frame (t)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(grid_out_np, cmap="viridis", origin="lower")
    axes[1].set_title("Target frame (t+1)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    out_path = output_dir / "grid_sample_train_0.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grid visualization to {out_path}")


if __name__ == "__main__":
    main()
