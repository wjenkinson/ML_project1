from pathlib import Path

import matplotlib.pyplot as plt
import torch

from grid_dataset import LammpsGridDataset
from train_cnn import SimpleFramePredictor


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


def main() -> None:
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cpu")

    # Load validation dataset for starting frame and ground truth comparison
    val_dataset = LammpsGridDataset(split="val", grid_size=(64, 64))
    if len(val_dataset) == 0:
        print("Validation dataset is empty; nothing to predict.")
        return

    # Use the first pair as a starting point
    start_in, start_target = val_dataset[0]

    # Load trained model
    model_path = output_dir / "simple_frame_predictor.pt"
    if not model_path.exists():
        print(f"Trained model not found at {model_path}; run train_cnn.py first.")
        return

    model = SimpleFramePredictor().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Number of future steps to roll out
    num_steps = 5

    # Save the starting input and its true next frame
    save_grid(output_dir / "predict_seq_step0_input.png", start_in, "Input t0 (val[0] input)")
    save_grid(output_dir / "predict_seq_step1_gt.png", start_target, "Ground truth t1 (val[0] target)")

    current = start_in.unsqueeze(0).to(device)  # shape (1, 1, H, W)

    for step in range(1, num_steps + 1):
        with torch.no_grad():
            pred = model(current)

        # Save the predicted grid for this step
        save_grid(
            output_dir / f"predict_seq_step{step}_pred.png",
            pred.squeeze(0),
            f"Predicted t{step}",
        )

        # Try to save a ground-truth frame for comparison if available
        if step < len(val_dataset):
            _, gt = val_dataset[step]
            save_grid(
                output_dir / f"predict_seq_step{step+1}_gt.png",
                gt,
                f"Ground truth t{step+1}",
            )

        # Feed prediction back in as the next input
        current = pred

    print(f"Saved prediction sequence images to {output_dir}")


if __name__ == "__main__":
    main()
