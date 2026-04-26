import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

CONFIG = {'title': 'Q18 Sales Data - Hierarchical Clustering', 'data_file': 'Sales_Data.csv', 'feature_cols': ['UnitsSold', 'Revenue', 'Profit'], 'drop_cols': [], 'label_col': None, 'pipeline': 'default'}


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CONFIG["feature_cols"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[CONFIG["feature_cols"]] = df[CONFIG["feature_cols"]].fillna(df[CONFIG["feature_cols"]].median())
    return df


def choose_clusters(linkage_matrix: np.ndarray, max_clusters: int = 8) -> int:
    distances = linkage_matrix[:, 2]
    if len(distances) < 3:
        return 2
    tail_count = min(max_clusters, len(distances))
    tail = distances[-tail_count:]
    gaps = np.diff(tail)
    if len(gaps) == 0:
        return 2
    return int(np.clip(np.argmax(gaps) + 2, 2, max_clusters))


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / CONFIG["data_file"]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = prepare_dataframe(df)

    features = df[CONFIG["feature_cols"]].copy()

    features.hist(figsize=(12, 8), bins=20)
    plt.suptitle("Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(features.iloc[:, 0], features.iloc[:, 1], alpha=0.6)
    plt.xlabel(CONFIG["feature_cols"][0])
    plt.ylabel(CONFIG["feature_cols"][1])
    plt.title("Raw Feature Relationship")
    plt.tight_layout()
    plt.show()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    linkage_matrix = linkage(X_scaled, method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrogram (Truncated)")
    plt.xlabel("Cluster leaves")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    n_clusters = choose_clusters(linkage_matrix)

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    agg_labels = agg.fit_predict(X_scaled)
    km_labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap="tab10", s=35)
    axes[0].set_title(f"Agglomerative Clusters (k={n_clusters})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="tab10", s=35)
    axes[1].set_title(f"K-Means Clusters (k={n_clusters})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.show()

    result = df.copy()
    result["HierarchicalCluster"] = agg_labels
    result["KMeansCluster"] = km_labels

    segment_profile = result.groupby("HierarchicalCluster")[CONFIG["feature_cols"]].mean().round(2)

    print(CONFIG["title"])
    print("=" * len(CONFIG["title"]))
    print(f"Rows used: {len(result)}")
    print(f"Features: {CONFIG['feature_cols']}")
    print(f"Selected clusters from dendrogram heuristic: {n_clusters}")

    if len(np.unique(agg_labels)) > 1:
        print(f"Agglomerative silhouette score: {silhouette_score(X_scaled, agg_labels):.4f}")
    if len(np.unique(km_labels)) > 1:
        print(f"K-Means silhouette score: {silhouette_score(X_scaled, km_labels):.4f}")

    print("\nHierarchical segment means:")
    print(segment_profile)


if __name__ == "__main__":
    main()
