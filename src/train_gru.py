from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_rnn import RnnSequenceDataset


class SimpleGruPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
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

    model = SimpleGruPredictor(input_size=train_dataset.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "simple_gru_predictor.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

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
            print(f"  Saved new best GRU model to {model_path}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
