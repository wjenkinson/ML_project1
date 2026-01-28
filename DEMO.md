# ML Project 1: LAMMPS Frame Prediction Demo

This document provides a visual walkthrough of the project pipeline, from raw LAMMPS data to trained model predictions.

## 1. Raw LAMMPS Data

We start with LAMMPS dump files containing particle positions over time. Here are visualizations of the first and last frames:

### Initial Frame (t=0)
![Initial Frame](output/visualization_dump.0.png)

### Frame at t=9000
![Later Frame](output/visualization_dump.9000.png)

## 2. Data Preprocessing

### Rasterization to 2D Grid
We convert the 3D particle positions into 2D density maps for processing with CNNs:

![Rasterized Grid](output/grid_sample_train_0.png)
*Left: Input frame (t), Right: Target frame (t+1)*

## 3. Training the CNN

We train a simple CNN to predict the next frame given the current one. The training progress looks like this:

```
Epoch 01 | train_loss=0.123456 | val_loss=0.098765
Epoch 02 | train_loss=0.098765 | val_loss=0.087654
...
```

## 4. Predictions

*[After training, we'll add visualizations of model predictions vs ground truth here.]*

## How to Reproduce

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate visualizations:
   ```bash
   python src/explore_data.py
   python src/visualize_grid.py
   ```

3. Train the model:
   ```bash
   python src/train_cnn.py
   ```
