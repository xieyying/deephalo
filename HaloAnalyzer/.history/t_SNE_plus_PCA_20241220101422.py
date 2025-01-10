import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import altair as alt

def plot_pca_2D(groups_data, group_labels):
    """
    Perform PCA on the provided data and plot the results using Altair.

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

    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    pca_df['Group'] = np.concatenate([[label] * len(group) for group, label in zip(groups_data, group_labels)])

    # Plot the PCA results using Altair
    chart = alt.Chart(pca_df).mark_circle(size=60).encode(
        x='PCA1',
        y='PCA2',
        color='Group',
        tooltip=['PCA1', 'PCA2', 'Group']
    ).properties(
        title='PCA Plot Showing Chemical Relationships'
    ).interactive()

    chart.show()

def plot_pca_3D(groups_data, group_labels):
    """
    Perform PCA on the provided data and plot the results using Altair.

    Parameters:
    groups_data (list of numpy.ndarray): List of high-dimensional vectors for each group.
    group_labels (list of str): List of labels for each group.

    Returns:
    None
    """
    # Combine all data
    all_data = np.vstack(groups_data)

    # Perform PCA
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(all_data)

    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Group'] = np.concatenate([[label] * len(group) for group, label in zip(groups_data, group_labels)])

    # Plot the PCA results using Altair
    chart = alt.Chart(pca_df).mark_circle(size=60).encode(
        x='PCA1',
        y='PCA2',
        color='Group',
        tooltip=['PCA1', 'PCA2', 'PCA3', 'Group']
    ).properties(
        title='PCA Plot Showing Chemical Relationships'
    ).interactive()

    chart.show()

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
    plot_pca_2D(groups_data, group_labels)
    plot_pca_3D(groups_data, group_labels)