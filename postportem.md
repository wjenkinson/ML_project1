### Reflections on Project 1

#### Plan

First we can discuss how it felt.
Then discuss what I think are my own personal weaknesses.
The goal is to identify the best next step.
Goals: to optimize learning, with a premium on ML for physics applications. Learn ML notions, learn the landscape, and get on the score board. This is promiscuous exploration.
Non-goals: Learn all the jargon, deep understanding of the toolboxes.

#### General vibe
Workflow with ML and physics simulations seems about 90% the same. Common elements: 
 - Large data sets -> Focus on pipeline automation using common data manip practices
 - Heavy use of parallelization protocols -> Scalability more important than cleverness
 - Integrated validation steps -> Immediate and continuous feedback assumed.

Key differences:
 - "Training", as a concept, doesn't exist for simulation models -> extra step in the total workflow
 - Physics-informed simulations usually have GT (e.g. discretizing NS or atom potentials) -> Hard to interpret the GT with ML models -> No obvious path to claim general prediction

#### Areas for improvement
 Need to be proactive to nurture intuition for known and unknowns in this field. Right now:
 - Increase personal visibility of the ML landscape.
 - Dense vocabulary in the field, work on filling the gap.
 - Future emphasis on time series, alternative data structures, physical constraints 

#### Overview of Future Work
Will build another project. Three neutral nets seem like appropriate progressions to the existing work: Recurrent, Graph and Physics-Informed Neural Nets (RNN, GNN & PINN)

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