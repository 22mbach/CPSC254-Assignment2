# CPSC254-Assignment2

CPSC 254 Assignment #2: Machine Learning Exercise (130 points). The assignment
covers supervised learning (regression and classification), unsupervised
learning (clustering), and reinforcement learning (policy iteration and a DQN
Snake agent), plus a written report.

## Status

| Part | Topic | File | Points | Status |
|------|-------|------|--------|--------|
| 1 | Linear regression | `linear_reg.py` | 20 | Done (output in `linear_reg_output.txt`, plot `linear_reg.png`) |
| 2 | MLP classification | `mlp_classifier.py` | 30 | Done (output in `mlp_classifier_output.txt`) |
| 3 | K-means clustering | `kmeans_clustering.py` | 30 | Done (plot `kmeans_clusters.png`) |
| 4.1 | Gridworld policy iteration | `gridworld_policy_iteration.py` | 20 | Done (output in `gridworld_output.txt`, screenshot `gridworld_converged.png`) |
| 4.2 | Snake DQN | `snake-ai-pytorch/` | 30 | Done (screenshots in `snake-ai-pytorch/`) |
| — | Report | `CPSC254_Assignment2_Report.pdf` | — | Done |

## Setup

Python 3.9+ is recommended. Install the dependencies used by Parts 1 to 4.1:

```powershell
pip install numpy pandas matplotlib scikit-learn
```

Part 4.2 additionally needs PyTorch, pygame and IPython (see below). Run every
command from the repository root unless noted; on Windows, use `py` in place
of `python` if `python` is not on your PATH.

## Part 1: Linear regression (`linear_reg.py`)

Requirements from the brief:

1. Load the small training set
   `X = [-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 2.7]`,
   `Y = [15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4]`.
2. Show a scatter plot with matplotlib.
3. Fit the data with scikit-learn's `LinearRegression()`.
4. Print the coefficient vector `[w0, w1, ..., wn]` and the fitted polynomial
   `y = w0 + w1 x + ... + wn x^n`.
5. Compute the training RMSE.
6. Load `GasProperties.csv`, split it 80/20 into train and test sets, solve
   for the weights with the normal equation `w = (XᵀX)⁻¹ Xᵀ y`, and report
   train and test RMSE.

`GasProperties.csv` (about 20 MB, 420,000 rows) has five numeric columns:
`T`, `P`, `TC`, `SV`, `Idx`.

The script fits both a straight line and a degree-2 polynomial
(`PolynomialFeatures` + `LinearRegression`); the data is U-shaped, so the
quadratic wins (training RMSE 0.63 vs. 4.75). For `GasProperties.csv` it uses
`T`, `P`, `TC`, `SV` as features and `Idx` as the target, giving a train RMSE
of 0.1365 and test RMSE of 0.1355.

```powershell
python linear_reg.py
```

Saves the scatter plot with both fits to `linear_reg.png` and opens it in a
window.

## Part 2: MLP classification (`mlp_classifier.py`)

Trains scikit-learn `MLPClassifier` models on the Iris dataset with an
80/20 stratified split:

- **(3) Baseline:** one hidden layer of 3 neurons (logistic, SGD,
  learning rate 0.001, 200 iterations), about 67% train and test accuracy.
- **(4) Tuning:** grid search with 5-fold CV over activation, solver,
  learning rate, alpha and max_iter, keeping the 1×3 architecture. Reaches
  97.5% train and 100% test accuracy.
- **(5) Larger networks:** compares several hidden-layer shapes and picks
  `(20, 10)`, at 98.3% train and 100% test accuracy.

Each step prints a short explanation of whether performance improved.

```powershell
python mlp_classifier.py
```

The output of a full run is saved in `mlp_classifier_output.txt`.

## Part 3: K-means clustering (`kmeans_clustering.py`)

1. Clusters Iris with `KMeans(n_clusters=3)` using only the four features
   (labels are ignored) and prints each cluster's RMSE to its centroid.
2. Validates cluster quality against the true classes: maps clusters to
   classes by majority vote, prints the confusion matrix, accuracy, ARI, NMI,
   homogeneity and completeness, and compares against an MLP classifier on the
   same 20% test split.

```powershell
python kmeans_clustering.py
```

Saves a side-by-side plot of the clusters and the true classes to
`kmeans_clusters.png` and opens it in a window.

## Part 4.1: Gridworld policy iteration (`gridworld_policy_iteration.py`)

Starter program from the course page: a 5×5 gridworld with a start state,
terminal states, an obstacle and a reward flag, plus an interactive matplotlib
viewer (buttons and sliders for gamma and noise) that shows the value
function as a heatmap.

Implemented:

- `policy_evaluation`: one Bellman expectation sweep,
  V(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [R(s,a,s') + γ V(s')], skipping terminals
  and obstacles.
- `policy_improvement`: computes Q(s,a) from V and picks the greedy action
  (ties broken consistently), returning `(pi, stable)`; `stable` is True when
  the new policy matches the previous one (optional `pi_old` argument).
- `policy_iteration`: repeats evaluation sweeps until V changes by less than
  `tol`, then improves, until the policy is stable.

With the defaults (γ = 0.95, noise = 0) it converges in 3 iterations. The
path from the start (4,0) goes up the left column and across row 1 to the
flag terminal (1,3), worth +15 (terminal +10 plus flag +5). With noise = 0.2
the start cell instead heads along the bottom row to (4,4).

```powershell
python gridworld_policy_iteration.py
```

## Part 4.2: Snake agent with a DQN (`snake-ai-pytorch/`)

A copy of Patrick Loeber's
[snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch)
(MIT license, see `snake-ai-pytorch/LICENSE`):

- `game.py`: the Snake environment (`SnakeGameAI`), built with pygame.
- `model.py`: the `Linear_QNet` network (11 inputs, 256 hidden, 3 outputs)
  and the `QTrainer` that applies the Bellman update.
- `agent.py`: the DQN agent and training loop.
- `helper.py`: live plot of score and mean score.
- `snake_game_human.py`: a version you can play yourself with the arrow keys.

```powershell
pip install torch pygame matplotlib ipython
cd snake-ai-pytorch
python agent.py
```

Run it from inside `snake-ai-pytorch/` so `game.py` can find `arial.ttf`.
Training keeps going until you close the window or press Ctrl+C, and the best
model is saved to `snake-ai-pytorch/model/model.pth`.

Screenshots used in the report: `Snake-game-running.png` (the game),
`training-plot.png` (score and mean score) and `Snake-game-terminal.png`
(terminal output).

## Report

`CPSC254_Assignment2_Report.pdf` has the answers and screenshots for every
part.

## Submission

Submit one report (Word or PDF) listing team members and each member's
percentage contribution (or "equal contribution"), with answers and
screenshots for each part, along with the program files above.
