from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from graph_dataset import LammpsGraphDataset


class SimpleGnnPredictor(nn.Module):
    """Simple 2-layer GNN for next-step particle position prediction.

    Node features: [x, y, z, type_id]
    Target: positions at t+1 for each particle.
    """

    def __init__(self, in_channels: int = 4, hidden_channels: int = 64, num_layers: int = 2) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.out_lin = nn.Linear(hidden_channels, 3)  # predict (x, y, z) at t+1

    def forward(self, data):  # type: ignore[override]
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        pred = self.out_lin(x)
        return pred


def train(
    epochs: int = 20,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    radius: float = 0.002,
) -> None:
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(0)

    train_dataset = LammpsGraphDataset(split="train", radius=radius)
    val_dataset = LammpsGraphDataset(split="val", radius=radius)

    if len(train_dataset) == 0:
        print("Train dataset is empty; nothing to train on.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleGnnPredictor(in_channels=4, hidden_channels=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "simple_gnn_predictor.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            preds = model(batch)  # (total_nodes_in_batch, 3)
            targets = batch.y.to(device)  # same shape

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
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch)
                targets = batch.y.to(device)

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
            print(f"  Saved new best GNN model to {model_path}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
