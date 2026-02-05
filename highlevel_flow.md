
```mermaid
flowchart TD

    %% Data source
    A[Raw LAMMPS dumps<br/>(data/dump.*.LAMMPS)] --> B[Data exploration<br/>src/explore_data.py<br/>Simulation GIF]

    %% Splits
    A --> C[Train/val split<br/>src/split_data.py<br/>data/splits/train/val_files.txt]

    %% Grid path
    C --> D[Grid dataset<br/>src/grid_dataset.py<br/>(t, t+1) density grids]
    D --> E[Grid visualization<br/>src/visualize_grid.py<br/>grid_sample_*.png]

    subgraph F[Grid-based sequence models]
        F1[Simple CNN<br/>src/train_cnn.py]
        F2[Vanilla RNN<br/>src/train_rnn.py]
        F3[GRU<br/>src/train_gru.py]
        F4[LSTM<br/>src/train_lstm.py]
    end

    D --> F

    %% Graph / GNN path
    C --> J[Graph dataset<br/>src/graph_dataset.py<br/>nodes=particles, radius edges]
    J --> K[GNN training<br/>src/train_gnn.py<br/>simple_gnn_predictor.pt]

    %% Unified prediction & evaluation
    F --> G[Predict sequences on val<br/>src/predict_sequence.py<br/>pred_seq_{tag}.pt]
    K --> G

    G --> H[Per-model videos<br/>src/make_video.py<br/>prediction_vs_gt_{tag}.gif]
    G --> I[Centerline sensitivity<br/>src/model_sensitivity.py<br/>model_sensitivity_master.png]

    %% Docs / reflection
    A --> L[Documentation & roadmap<br/>README.md, DEMO.md, roadmap.md, postmortem.md]
    H --> L
    I --> L
```