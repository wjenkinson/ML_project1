### Reflections on Project 1

#### General thoughts
Key similarities between ML and physics simulations workflows: 
 - Large data sets → Focus on pipeline automation using common data manip practices
 - Heavy use of parallelization protocols → Scalability more important than cleverness
 - Integrated validation steps → Immediate and continuous feedback.

Key differences:
 - "Training", as a concept, doesn't exist for simulation models → extra step in the total workflow
 - Physics-informed simulations usually have GT (e.g. discretizing conservation equations) → Absense of GT with ML models

#### Areas for personal improvement
 - Increase visibility of the ML landscape.
 - Dense vocabulary in the field, work on filling the gap.
 - Future emphasis on time series, alternative data structures, physical constraints 

#### Overview of Future Work
Three neutral nets seem like appropriate progressions to the existing work: Recurrent, Graph and Physics-Informed Neural Nets (RNN, GNN & PINN)

##### RNNs
Designed to capture temporal dependencies.

New concepts:
 - backpropagation, variants like LSTM and GRU

##### GNNs
Captures relational structures in data, an essential feature of atomic interactions.

New concepts:
 - graph representation

##### PINNs
Build physics constraints into the learning process e.g. to enforce conservation laws.

New concepts:
 - physics-based loss function

#### Next step
Address some gaps left by the initial Project 1 roadmap.

Once gaps addressed, move on to PINNs in Project 2.