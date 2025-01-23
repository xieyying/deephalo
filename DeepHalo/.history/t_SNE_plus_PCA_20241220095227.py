import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def plot_pca(groups_data, group_labels):
    """
    Perform PCA on the provided data and plot the results.

    Parameters:
    groups_data (list of numpy.ndarray): List of high-dimensional vectors for each group.
    group_labels (list of str): List of labels for each group.

    Returns:
    None
    """
    # Combine all data
    all_data = np.vstack(groups_data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(all_data)

    # Split the results back into the original groups
    split_indices = np.cumsum([len(group) for group in groups_data])
    pca_groups = np.split(pca_results, split_indices[:-1])

    # Plot the PCA results
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'yellow', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for pca_group, label, color in zip(pca_groups, group_labels, colors):
        plt.scatter(pca_group[:, 0], pca_group[:, 1], label=label, color=color)

    plt.legend()
    plt.title('PCA Plot Showing Chemical Relationships')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

if __name__ == '__main__':
    validation_data = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\train_and_val\validation_data.csv'
    df = pd.read_csv(validation_data)

    # Define the feature list
    feature_list = [
        'p0_int',
        'p1_int',
        'p2_int',
        'p3_int',
        'p4_int',
        'm1_m0',
        'm2_m1',
    ]

    # Extract data for each group
    groups_data = [df[df['group'] == i][feature_list].values for i in range(8)]
    group_labels = [f'Group {i}' for i in range(8)]

    # Plot PCA results
    plot_pca(groups_data, group_labels)