from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from grid_dataset import LammpsGridDataset


class SimpleFramePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def train(
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
) -> None:
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    # For this project, always run on CPU to avoid CUDA compatibility issues.
    device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(0)

    train_dataset = LammpsGridDataset(split="train", grid_size=(64, 64))
    val_dataset = LammpsGridDataset(split="val", grid_size=(64, 64))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleFramePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "simple_frame_predictor.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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

                outputs = model(inputs)
                loss = criterion(outputs, targets)

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
            print(f"  Saved new best model to {model_path}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
