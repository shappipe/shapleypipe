"""
Clustering analysis and metrics computation
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(shapley_matrix):
    """
    Compute pairwise similarity matrix between operators

    Args:
        shapley_matrix: Shape (n_operators, n_datasets, n_positions)

    Returns:
        similarity_matrix: Shape (n_operators, n_operators)
    """
    # Flatten each operator's profile: (n_datasets * n_positions,)
    n_operators = shapley_matrix.shape[0]
    operator_profiles = shapley_matrix.reshape(n_operators, -1)

    # Compute correlation-based similarity
    similarity_matrix = np.corrcoef(operator_profiles)

    return similarity_matrix


def perform_clustering(shapley_matrix, n_clusters=5):
    """
    Perform hierarchical category_validation_experiment on operators

    Args:
        shapley_matrix: Shape (n_operators, n_datasets, n_positions)
        n_clusters: Number of clusters to form

    Returns:
        linkage_matrix: Hierarchical category_validation_experiment linkage
        cluster_labels: Cluster assignment for each operator
    """
    n_operators = shapley_matrix.shape[0]
    operator_profiles = shapley_matrix.reshape(n_operators, -1)

    # Hierarchical category_validation_experiment with Ward linkage
    linkage_matrix = linkage(operator_profiles, method='ward')

    # Cut dendrogram to form clusters
    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

    return linkage_matrix, cluster_labels


def compute_metrics(similarity_matrix, true_labels, predicted_labels):
    """
    Compute evaluation metrics for category_validation_experiment quality

    Args:
        similarity_matrix: Operator similarity matrix, shape (n_operators, n_operators, n_pipeline_length)
        true_labels: Ground truth category labels            (n_operators,)
        predicted_labels: Predicted cluster labels           (n_operators,)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    n_operators = len(true_labels)

    # Within-category correlation
    within_corr = []
    for label in np.unique(true_labels):
        indices = np.where(true_labels == label)[0]
        if len(indices) > 1:
            sub_matrix = similarity_matrix[np.ix_(indices, indices)]
            # Get upper triangle (exclude diagonal)
            mask = np.triu(np.ones_like(sub_matrix, dtype=bool), k=1)
            within_corr.extend(sub_matrix[mask])

    rho_within_mean = np.mean(within_corr)
    rho_within_std = np.std(within_corr)

    # Between-category correlation
    between_corr = []
    unique_labels = np.unique(true_labels)
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1:]:
            idx1 = np.where(true_labels == label1)[0]
            idx2 = np.where(true_labels == label2)[0]
            sub_matrix = similarity_matrix[np.ix_(idx1, idx2)]
            between_corr.extend(sub_matrix.flatten())

    rho_between_mean = np.mean(between_corr)
    rho_between_std = np.std(between_corr)

    # Silhouette score
    operator_profiles = 1 - similarity_matrix  # Convert to distance
    np.fill_diagonal(operator_profiles, 0)  # Set diagonal to 0

    silhouette = silhouette_score(
        operator_profiles,
        true_labels,
        metric='precomputed',
        random_state=42
    )

    # Adjusted Rand Index
    ari = adjusted_rand_score(true_labels, predicted_labels)

    metrics = {
        'rho_within_mean': rho_within_mean,
        'rho_within_std': rho_within_std,
        'rho_between_mean': rho_between_mean,
        'rho_between_std': rho_between_std,
        'silhouette_score': silhouette,
        'adjusted_rand_index': ari,
        'separation_ratio': rho_within_mean / rho_between_mean if rho_between_mean > 0 else np.inf
    }

    return metrics
