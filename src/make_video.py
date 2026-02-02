from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import torch

from grid_dataset import LammpsGridDataset
from train_cnn import SimpleFramePredictor


def main() -> None:
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cpu")

    # Load validation dataset as the ground-truth sequence
    val_dataset = LammpsGridDataset(split="val", grid_size=(64, 64))
    if len(val_dataset) == 0:
        print("Validation dataset is empty; nothing to visualize.")
        return

    # Build ground-truth frames: t0 input, then all targets
    start_in, _ = val_dataset[0]
    gt_frames = [start_in]
    for i in range(len(val_dataset)):
        _, gt = val_dataset[i]
        gt_frames.append(gt)

    # Load trained model
    model_path = output_dir / "simple_frame_predictor.pt"
    if not model_path.exists():
        print(f"Trained model not found at {model_path}; run train_cnn.py first.")
        return

    model = SimpleFramePredictor().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Build predicted sequence by autoregressively rolling out from t0
    pred_frames = []
    current = start_in.unsqueeze(0).to(device)  # shape (1, 1, H, W)
    with torch.no_grad():
        for _ in range(len(gt_frames) - 1):
            pred = model(current)
            pred_frames.append(pred.squeeze(0).cpu())
            current = pred

    # Convert tensors to numpy arrays
    gt_imgs = [f.squeeze(0).detach().cpu().numpy() for f in gt_frames[1:]]  # t1..tN
    pred_imgs = [f.squeeze(0).detach().cpu().numpy() for f in pred_frames]  # same length

    num_frames = min(len(gt_imgs), len(pred_imgs))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(gt_imgs[0], cmap="viridis", origin="lower")
    axes[0].set_title("Ground truth")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(pred_imgs[0], cmap="viridis", origin="lower")
    axes[1].set_title("Prediction")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()

    def update(frame_idx: int):
        im0.set_data(gt_imgs[frame_idx])
        im1.set_data(pred_imgs[frame_idx])
        axes[0].set_title(f"Ground truth (step {frame_idx + 1})")
        axes[1].set_title(f"Prediction (step {frame_idx + 1})")
        return [im0, im1]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=300,
        blit=True,
    )

    gif_path = output_dir / "prediction_vs_gt.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=4))
    plt.close(fig)

    print(f"Saved side-by-side GIF to {gif_path}")


if __name__ == "__main__":
    main()
