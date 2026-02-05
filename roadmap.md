### Project 1: Naive Frame Prediction

#### **Blurb: Scope and Non-Scope**
**Scope:** Build a minimal model to predict the next frame in a simulation dataset. Focus on understanding the mechanics of training an ML model, data handling, and evaluation. The output is purely for learning; visual quality is the primary criterion.

**Non-Scope:** Hyperparameter tuning, deployment, user interface, interpretability tools, or integration with other systems. Physical plausibility is not a priority.

---

#### ** (COMPLETE) Phase 1: Data Preparation**
**Description:** Prepare the dataset for training by inspecting, splitting, and formatting the data.

**Tasks:**
- Inspect the dataset: Visualize frames to understand structure and content.
- Split the dataset: Divide into training (80%) and validation (20%) sets.
- Format the data: Convert frames into tensors compatible with PyTorch or TensorFlow.

**Success Criteria:**
- Dataset is successfully loaded and visualized.
- Data is split into training and validation sets without errors.
- Data is formatted correctly for model input.

**Failure Criteria:**
- Data cannot be loaded or visualized due to structural issues.
- Splitting or formatting introduces errors that prevent training.

---

#### ** (COMPLETE) Phase 2: Model Selection and Prototyping**
**Description:** Implement a simple model architecture to predict the next frame.

**Tasks:**
- Select a baseline model: Start with a 3D CNN or LSTM.
- Implement the model: Use a framework like PyTorch or TensorFlow.
- Train the model: Use the training split with a basic loss function (e.g., MSE).

**Success Criteria:**
- Model is implemented and runs without errors.
- Initial training completes without crashing.
- Training loss decreases over epochs.

**Failure Criteria:**
- Model fails to train due to architectural or data issues.
- Training loss does not decrease, indicating no learning.

---

#### ** (COMPLETE) Phase 3: Evaluation and Learning**
**Description:** Evaluate the model’s output and reflect on the process.

**Tasks:**
- Generate predictions: Run the model on the validation set.
- Visual inspection: Compare predicted frames to ground truth visually.
- Document insights: Note what worked, what didn’t, and why.

**Success Criteria:**
- Predictions are visually inspectable and provide clear learning insights.
- Documentation captures key lessons and trade-offs.

**Failure Criteria:**
- Predictions are completely incoherent (e.g., noise or artifacts dominate).
- No meaningful insights are gained from the process.

---

#### **Phase 4: Postmortem (Project Retrospective)**
**Description:** Review the project to solidify understanding and identify next steps.

**Tasks:**
- Summarize findings: What did you learn about ML workflows, model choices, and data handling?
- Identify gaps: Where did assumptions fail or knowledge fall short?
- Plan next steps: What would you change or explore further in a follow-up project?

**Success Criteria:**
- Clear, actionable insights are documented.
- Next steps are defined for continued learning.

**Failure Criteria:**
- No actionable insights or next steps are identified.

---

### Project 1 Extension: RNN and GNN

#### (COMPLETE) Phase 5: Extension to RNN
**Description:** Implement an RNN to capture temporal dependencies in the data.

**Tasks:**
- **Model Selection:** Choose an LSTM or GRU architecture.
- **Implementation:** Use PyTorch to implement the RNN.
- **Training:** Train the model on the same dataset used for the CNN.

**Success Criteria:**
- RNN model trains without errors.
- Training loss decreases over epochs.
- Predictions are visually inspectable and provide clear learning insights.

**Failure Criteria:**
- RNN implementation fails to improve on CNN baseline or crashes during training.
- Predictions are completely incoherent (e.g., noise or artifacts dominate).

**Resources:**
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

#### (COMPLETE) Phase 6: Extension to GNN
**Description:** Implement a GNN to capture relational structures in the data.

**Tasks:**
- **Data Representation:** Convert atomic coordinates and velocities into a graph structure.
- **Model Selection:** Choose a GNN architecture (e.g., GCN, GraphSAGE).
- **Implementation:** Use a library like PyTorch Geometric to implement the GNN.
- **Training:** Train the model on the graph-structured data.

**Success Criteria:**
- GNN model trains without errors.
- Training loss decreases over epochs.
- Predictions respect relational structures and provide clear learning insights.

**Failure Criteria:**
- GNN implementation fails to capture relational structures or crashes during training.
- Predictions are completely incoherent (e.g., noise or artifacts dominate).

**Resources:**
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
- [Graph Neural Networks for Molecular Dynamics](https://arxiv.org/abs/2003.02035)

---

#### Phase 7: Post Mortem 2
**Description:** Review the extended project to solidify understanding and identify next steps.

**Tasks:**
- **Summarize Findings:** What did you learn about RNNs and GNNs?
- **Identify Gaps:** Where did assumptions fail or knowledge fall short?
- **Plan Next Steps:** What would you change or explore further in a follow-up project?

**Success Criteria:**
- Clear, actionable insights are documented.
- Next steps are defined for continued learning.

**Failure Criteria:**
- No actionable insights or next steps are identified.

---