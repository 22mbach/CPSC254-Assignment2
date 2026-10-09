"""
CPSC 254 Assignment #2, Part 3: Unsupervised learning for clustering.

kmeans_clustering.py
  (1) Clusters the Iris flower dataset with K-means (K=3) using scikit-learn's
      KMeans, ignoring the label column, and computes the RMSE of each cluster
      with respect to its centroid.
  (2) Compares the clustering with the actual classes and with the results of
      an MLPClassifier (as in mlp_classifier.py) to validate cluster quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
)

RANDOM_STATE = 42
K = 3


def cluster_rmse(X, labels, centroids):
    """RMSE of each cluster: sqrt(mean squared Euclidean distance to its centroid)."""
    rmse = []
    for k, c in enumerate(centroids):
        points = X[labels == k]
        sq_dist = np.sum((points - c) ** 2, axis=1)
        rmse.append(np.sqrt(np.mean(sq_dist)))
    return np.array(rmse)


def map_clusters_to_classes(clusters, y):
    """Map each cluster id to the true class that occurs most often in it."""
    mapping = {}
    for k in np.unique(clusters):
        mapping[k] = np.bincount(y[clusters == k]).argmax()
    return mapping


def main():
    # ---------------------------------------------------------------
    # Load the Iris dataset; only the features are used for clustering.
    # ---------------------------------------------------------------
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names
    print("Iris dataset:", X.shape[0], "samples,", X.shape[1], "features")
    print("Features:", ", ".join(iris.feature_names))

    # ---------------------------------------------------------------
    # (1) K-means with K=3 on the unlabeled data
    # ---------------------------------------------------------------
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    print("\n=== (1) K-means clustering (K=3) ===")
    rmse = cluster_rmse(X, clusters, centroids)
    for k in range(K):
        size = np.sum(clusters == k)
        print(f"Cluster {k}: size = {size:3d}, "
              f"centroid = {np.round(centroids[k], 3)}, RMSE = {rmse[k]:.4f}")
    overall_rmse = np.sqrt(kmeans.inertia_ / len(X))
    print(f"Overall RMSE (all points to their centroid) = {overall_rmse:.4f}")
    print(f"Inertia (sum of squared distances) = {kmeans.inertia_:.4f}")

    # ---------------------------------------------------------------
    # (2) Cluster quality validation against the actual classes
    # ---------------------------------------------------------------
    print("\n=== (2) Cluster quality validation ===")
    mapping = map_clusters_to_classes(clusters, y)
    for k, c in mapping.items():
        print(f"Cluster {k} -> {class_names[c]}")
    y_cluster = np.array([mapping[k] for k in clusters])

    cm = confusion_matrix(y, y_cluster)
    print("\nConfusion matrix (rows = actual class, cols = cluster's class):")
    print(f"{'':>12}" + "".join(f"{n:>12}" for n in class_names))
    for name, row in zip(class_names, cm):
        print(f"{name:>12}" + "".join(f"{v:>12d}" for v in row))

    km_acc_all = accuracy_score(y, y_cluster)
    print(f"\nK-means accuracy (clusters mapped to classes) = {km_acc_all * 100:.2f}%")
    print(f"Adjusted Rand Index   = {adjusted_rand_score(y, clusters):.4f}")
    print(f"Normalized Mutual Info = {normalized_mutual_info_score(y, clusters):.4f}")
    print(f"Homogeneity           = {homogeneity_score(y, clusters):.4f}")
    print(f"Completeness          = {completeness_score(y, clusters):.4f}")

    # Comparison with the supervised MLP classifier from mlp_classifier.py,
    # trained on the same 80/20 split.
    X_train, X_test, y_train, y_test, _, c_test = train_test_split(
        X, y, clusters, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation="relu",
                        solver="adam", max_iter=2000,
                        random_state=RANDOM_STATE)
    mlp.fit(X_train, y_train)
    mlp_train_acc = accuracy_score(y_train, mlp.predict(X_train))
    mlp_test_acc = accuracy_score(y_test, mlp.predict(X_test))
    km_test_acc = accuracy_score(y_test, [mapping[k] for k in c_test])

    print("\nK-means vs. MLP classifier (same 20% test set):")
    print(f"{'Method':<32}{'Accuracy':>10}")
    print(f"{'MLP (supervised), train':<32}{mlp_train_acc * 100:>9.2f}%")
    print(f"{'MLP (supervised), test':<32}{mlp_test_acc * 100:>9.2f}%")
    print(f"{'K-means (unsupervised), test':<32}{km_test_acc * 100:>9.2f}%")
    print(f"{'K-means (unsupervised), all':<32}{km_acc_all * 100:>9.2f}%")

    print("\nDiscussion:")
    print("- One cluster matches Iris setosa exactly; setosa is linearly")
    print("  separable from the other two species.")
    print("- Versicolor and virginica overlap in feature space, so K-means,")
    print("  which only sees distances, mixes some of them. The MLP uses the")
    print("  labels during training, so it learns that boundary and scores higher.")
    print(f"- Without any labels, K-means still recovers {km_acc_all * 100:.0f}% of the")
    print("  true class structure, showing the clusters align well with the classes.")

    # ---------------------------------------------------------------
    # Visualization: clusters vs. actual classes (petal features)
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(X[:, 2], X[:, 3], c=clusters, cmap="viridis", s=30)
    axes[0].scatter(centroids[:, 2], centroids[:, 3], c="red", marker="X",
                    s=200, label="Centroids")
    axes[0].set_title("K-means clusters (K=3)")
    axes[0].legend()
    for c, name in enumerate(class_names):
        pts = X[y == c]
        axes[1].scatter(pts[:, 2], pts[:, 3], s=30, label=name)
    axes[1].set_title("Actual classes")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel(iris.feature_names[2])
    axes[0].set_ylabel(iris.feature_names[3])
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=120)
    print("\nPlot saved to kmeans_clusters.png")
    plt.show()


if __name__ == "__main__":
    main()
