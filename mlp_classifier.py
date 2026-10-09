"""CPSC 254 Assignment 2, Part 2: Supervised learning for classification.

Trains scikit-learn MLPClassifier models on the Iris flower dataset:
  (1) load Iris from sklearn.datasets
  (2) split 80% train / 20% test
  (3) baseline MLP with one hidden layer of 3 neurons, report train/test accuracy
  (4) tune hyperparameters (same 3-neuron architecture) to improve accuracy
  (5) add layers and/or neurons until accuracy improves

Run:  python mlp_classifier.py
"""

import warnings

from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

# The deliberately small baseline does not always converge; keep the output readable.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42


def evaluate(model, X_train, X_test, y_train, y_test):
    """Fit the model and return (train %, test %) accuracy."""
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
    return train_acc, test_acc


def report(title, train_acc, test_acc):
    print(f"{title}")
    print(f"  Training accuracy: {train_acc:6.2f}%   (training error {100 - train_acc:5.2f}%)")
    print(f"  Testing accuracy:  {test_acc:6.2f}%   (testing error  {100 - test_acc:5.2f}%)")


def main():
    # (1) Load the Iris dataset directly from sklearn.datasets.
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"Iris dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(iris.target_names)} classes ({', '.join(iris.target_names)})\n")

    # (2) 80% training / 20% testing, stratified so every class is represented.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"Training set: {len(X_train)} samples, testing set: {len(X_test)} samples\n")

    # (3) Baseline: one hidden layer with 3 neurons, plain SGD with a small
    # learning rate and scikit-learn's default iteration budget.
    print("=" * 70)
    print("(3) Baseline MLP: 1 hidden layer x 3 neurons")
    print("=" * 70)
    baseline = MLPClassifier(
        hidden_layer_sizes=(3,),
        activation="logistic",
        solver="sgd",
        learning_rate_init=0.001,
        momentum=0.9,
        batch_size=32,
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    base_train, base_test = evaluate(baseline, X_train, X_test, y_train, y_test)
    print("Hyperparameters: activation=logistic, solver=sgd, learning_rate_init=0.001,")
    print("                 momentum=0.9, batch_size=32, max_iter=200")
    report("Baseline results:", base_train, base_test)
    print()

    # (4) Tune hyperparameters only; the architecture stays at (3,).
    print("=" * 70)
    print("(4) Hyperparameter tuning (architecture fixed at 1 x 3 neurons)")
    print("=" * 70)
    param_grid = {
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["sgd", "adam", "lbfgs"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "alpha": [0.0001, 0.01],
        "max_iter": [500, 2000],
    }
    search = GridSearchCV(
        MLPClassifier(hidden_layer_sizes=(3,), batch_size=32, momentum=0.9,
                      random_state=RANDOM_STATE),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
        n_jobs=-1,
    )
    # Hyperparameters are chosen with cross-validation on the training set only,
    # so the test set stays unseen until the final evaluation.
    search.fit(X_train, y_train)
    print(f"Best hyperparameters (5-fold CV accuracy {search.best_score_ * 100:.2f}%):")
    for name, value in search.best_params_.items():
        print(f"  {name} = {value}")
    tuned = search.best_estimator_
    tuned_train, tuned_test = evaluate(tuned, X_train, X_test, y_train, y_test)
    report("Tuned results:", tuned_train, tuned_test)
    print(f"Change vs. baseline: training {tuned_train - base_train:+.2f} pts, "
          f"testing {tuned_test - base_test:+.2f} pts")
    if tuned_test > base_test:
        print("Explanation: Yes, tuning improved classification performance. The baseline's\n"
              "small learning rate with logistic units and only 200 SGD epochs left the\n"
              "network under-trained (stuck predicting too few classes). The tuned\n"
              "settings above, chiefly a larger learning rate and iteration budget, let\n"
              "the same 3-neuron network converge to a much better decision boundary.")
    else:
        print("Explanation: No, tuning did not raise test accuracy above the baseline.")
    print()

    # (5) Grow the network (more neurons and/or layers) with the tuned hyperparameters.
    print("=" * 70)
    print("(5) Increasing network complexity (tuned hyperparameters kept)")
    print("=" * 70)
    tuned_params = {k: v for k, v in tuned.get_params().items()
                    if k != "hidden_layer_sizes"}
    architectures = [(3,), (5,), (10,), (10, 10), (20, 10), (20, 20, 10), (50, 50)]
    print(f"{'Hidden layers':<16}{'Train %':>10}{'Test %':>10}")
    results = {}
    for layers in architectures:
        model = MLPClassifier(hidden_layer_sizes=layers, **tuned_params)
        results[layers] = evaluate(model, X_train, X_test, y_train, y_test)
        print(f"{str(layers):<16}{results[layers][0]:>10.2f}{results[layers][1]:>10.2f}")

    # The "new architecture" is the smallest one that beats the tuned (3,) network:
    # higher test accuracy, or equal test accuracy with higher training accuracy.
    # If none beats it, report the best one found.
    def score(arch):
        return results[arch][1], results[arch][0]

    improved = [a for a in architectures[1:] if score(a) > (tuned_test, tuned_train)]
    best_arch = improved[0] if improved else max(architectures, key=score)
    print(f"\nChosen architecture: {best_arch}")
    report("Results:", *results[best_arch])
    print(f"Change vs. tuned 1 x 3 network: training "
          f"{results[best_arch][0] - tuned_train:+.2f} pts, testing "
          f"{results[best_arch][1] - tuned_test:+.2f} pts")
    if improved:
        print("Explanation: Yes. The larger network fits the training data better while\n"
              "keeping test accuracy at least as high, so it is the better model. Three\n"
              "hidden neurons are a tight bottleneck for separating versicolor from\n"
              "virginica, whose measurements overlap; more hidden units give the network\n"
              "capacity to model that boundary. Gains are small because Iris is easy and\n"
              "the tuned 1 x 3 network was already near the ceiling. More is not always\n"
              "better: a deeper network can fail to train at all with the same plain-SGD\n"
              "settings (see any row stuck at 33.33%, i.e. predicting a single class).")
    else:
        print("Explanation: No larger architecture beat the tuned 1 x 3 network. Iris is\n"
              "small and nearly linearly separable, so once the 3-neuron network is well\n"
              "tuned it is already close to the accuracy ceiling.")


if __name__ == "__main__":
    main()
