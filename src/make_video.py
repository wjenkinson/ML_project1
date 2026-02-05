from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import torch


def main() -> None:
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    seq_files = sorted(output_dir.glob("pred_seq_*.pt"))
    if not seq_files:
        print("No prediction sequence files found; run predict_sequence.py first.")
        return

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

        num_frames = min(len(gt_frames), len(pred_frames))
        if num_frames == 0:
            print(f"{seq_path} contains empty sequences; skipping.")
            continue

        # Convert tensors to numpy arrays
        gt_imgs = [f.squeeze(0).detach().cpu().numpy() for f in gt_frames[:num_frames]]
        pred_imgs = [f.squeeze(0).detach().cpu().numpy() for f in pred_frames[:num_frames]]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        im0 = axes[0].imshow(gt_imgs[0], cmap="viridis", origin="lower")
        axes[0].set_title("Ground truth")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        im1 = axes[1].imshow(pred_imgs[0], cmap="viridis", origin="lower")
        axes[1].set_title(f"Prediction ({tag})")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.tight_layout()

        def update(frame_idx: int):
            im0.set_data(gt_imgs[frame_idx])
            im1.set_data(pred_imgs[frame_idx])
            axes[0].set_title(f"Ground truth (step {frame_idx + 1})")
            axes[1].set_title(f"Prediction ({tag}) (step {frame_idx + 1})")
            return [im0, im1]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=300,
            blit=True,
        )

        gif_path = output_dir / f"prediction_vs_gt_{tag}.gif"
        anim.save(gif_path, writer=animation.PillowWriter(fps=4))
        plt.close(fig)

        print(f"Saved side-by-side GIF for model tag '{tag}' to {gif_path}")


if __name__ == "__main__":
    main()
