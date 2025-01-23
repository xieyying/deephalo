import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

def plot_tsne_pca(training_data, prediction_data, abaucin_data):
    """
    Perform t-SNE and PCA on the provided data and plot the results.

    Parameters:
    training_data (numpy.ndarray): High-dimensional vectors for the training dataset.
    prediction_data (numpy.ndarray): High-dimensional vectors for the prediction set.
    abaucin_data (numpy.ndarray): High-dimensional vector for abaucin.

    Returns:
    None
    """
    # Combine all data
    all_data = np.vstack([training_data, prediction_data, abaucin_data])

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_data)

    # Split the results back into the original groups
    training_tsne = tsne_results[:len(training_data)]
    prediction_tsne = tsne_results[len(training_data):len(training_data) + len(prediction_data)]
    abaucin_tsne = tsne_results[len(training_data) + len(prediction_data):]

    # Perform PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(all_data)

    # Split the results back into the original groups
    training_pca = pca_results[:len(training_data)]
    prediction_pca = pca_results[len(training_data):len(training_data) + len(prediction_data)]
    abaucin_pca = pca_results[len(training_data) + len(prediction_data):]

    # Plot the t-SNE results
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.scatter(training_tsne[:, 0], training_tsne[:, 1], color='blue', label='Training Dataset')
    plt.scatter(prediction_tsne[:, 0], prediction_tsne[:, 1], color='red', label='Prediction Set')
    plt.scatter(abaucin_tsne[:, 0], abaucin_tsne[:, 1], color='yellow', label='Abaucin', edgecolor='black', s=100)
    plt.legend()
    plt.title('t-SNE Plot Showing Chemical Relationships')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Plot the PCA results
    plt.subplot(1, 2, 2)
    plt.scatter(training_pca[:, 0], training_pca[:, 1], color='blue', label='Training Dataset')
    plt.scatter(prediction_pca[:, 0], prediction_pca[:, 1], color='red', label='Prediction Set')
    plt.scatter(abaucin_pca[:, 0], abaucin_pca[:, 1], color='yellow', label='Abaucin', edgecolor='black', s=100)
    plt.legend()
    plt.title('PCA Plot Showing Chemical Relationships')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    plt.show()

if __name__ == '__main__':
    # Create some random data for demonstration purposes
    # training_data = np.random.rand(100, 50)  # 100 samples, 50 features
    # prediction_data = np.random.rand(30, 50)  # 30 samples, 50 features
    # abaucin_data = np.random.rand(2, 50)  # 1 sample, 50 features
    # print(abaucin_data)
    validation_data = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\train_and_val\validation_data.csv'
    df = pd.read_csv(validation_data)
    df1 = df[df['group'] == 1]
    df2 = df[df['group'] == 2]
    df3 = df[df['group'] == 3]
    df4 = df[df['group'] == 4]
    df5 = df[df['group'] == 5]
    df6 = df[df['group'] == 6]
    df7 = df[df['group'] == 7]
    
    feature_list=[  'p0_int',
                    'p1_int',
                    'p2_int',
                    'p3_int',
                    'p4_int',
                    # 'p5_int',
                    # 'p6_int',
                    'm1_m0',
                    'm2_m1',
                    # 'mz_0',
    ]
    # plot_tsne_pca(training_data, prediction_data, abaucin_data)