# CPSC254-Assignment2

CPSC 254 Assignment #2: Machine Learning Exercise (130 points). The assignment
covers supervised learning (regression and classification), unsupervised
learning (clustering), and reinforcement learning (policy iteration and a DQN
Snake agent), plus a written report.

## Status

| Part | Topic | File | Points | Status |
|------|-------|------|--------|--------|
| 1 | Linear regression | `linear_reg.py` | 20 | Done |
| 2 | MLP classification | `mlp_classifier.py` | 30 | Done (sample run in `mlp_classifier_output.txt`) |
| 3 | K-means clustering | `kmeans_clustering.py` | 30 | Done |
| 4.1 | Gridworld policy iteration | `gridworld_policy_iteration.py` | 20 | Starter file only; the three TODO functions are still empty |
| 4.2 | Snake DQN | external repo | 30 | Not started (screenshots and answers go in the report) |
| — | Report | Word or PDF | — | Not started |

## Setup

Python 3.9+ is recommended. Install the dependencies used by Parts 1 to 4.1:

```bash
pip install numpy matplotlib scikit-learn
```

Part 4.2 additionally needs PyTorch and pygame (see below).

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

```bash
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

```bash
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

```bash
python kmeans_clustering.py
```

Saves a side-by-side plot of the clusters and the true classes to
`kmeans_clusters.png` and opens it in a window.

## Part 4.1: Gridworld policy iteration (`gridworld_policy_iteration.py`)

Starter program from the course page: a 5×5 gridworld with a start state,
terminal states, an obstacle and a reward flag, plus an interactive matplotlib
viewer (buttons and sliders for gamma and noise) that shows the value
function as a heatmap.

Still to implement:

- `policy_evaluation`: one sweep computing V(s) under policy π.
- `policy_improvement`: greedy update of π from V(s), returning `(pi, stable)`.
- `policy_iteration`: alternate evaluation and improvement until π is stable.

```bash
python gridworld_policy_iteration.py
```

## Part 4.2: Snake agent with a DQN

Uses Patrick Loeber's
[snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch). Per
the brief, that code is not committed here; only screenshots and answers go
in the report.

```bash
git clone https://github.com/patrickloeber/snake-ai-pytorch
cd snake-ai-pytorch
pip install torch pygame matplotlib ipython
python agent.py
```

The report needs screenshots of the game, the training plot (score and mean
score) and the terminal output, plus answers on the state representation,
action space, reward function and what each DQN output represents.

## Submission

Submit one report (Word or PDF) listing team members and each member's
percentage contribution (or "equal contribution"), with answers and
screenshots for each part, along with the program files above. Do not upload
downloaded packages.
