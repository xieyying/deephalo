# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# # Example data: high-dimensional vectors for training, prediction, and abaucin
# training_data = np.random.rand(100, 50)  # 100 samples, 50 features
# prediction_data = np.random.rand(30, 50)  # 30 samples, 50 features
# abaucin_data = np.random.rand(1, 50)  # 1 sample, 50 features

# # Combine all data
# all_data = np.vstack([training_data, prediction_data, abaucin_data])

# # Perform t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(all_data)

# # Split the results back into the original groups
# training_tsne = tsne_results[:100]
# prediction_tsne = tsne_results[100:130]
# abaucin_tsne = tsne_results[130:]

# # Plot the t-SNE results
# plt.figure(figsize=(10, 8))
# plt.scatter(training_tsne[:, 0], training_tsne[:, 1], color='blue', label='Training Dataset')
# plt.scatter(prediction_tsne[:, 0], prediction_tsne[:, 1], color='red', label='Prediction Set')
# plt.scatter(abaucin_tsne[:, 0], abaucin_tsne[:, 1], color='yellow', label='Abaucin', edgecolor='black', s=100)
# plt.legend()
# plt.title('t-SNE Plot Showing Chemical Relationships')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.show()


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Example data: high-dimensional vectors for training, prediction, and abaucin
training_data = np.random.rand(100, 50)  # 100 samples, 50 features
prediction_data = np.random.rand(30, 50)  # 30 samples, 50 features
abaucin_data = np.random.rand(1, 50)  # 1 sample, 50 features

# Combine all data
all_data = np.vstack([training_data, prediction_data, abaucin_data])

# Perform PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(all_data)

# Split the results back into the original groups
training_pca = pca_results[:100]
prediction_pca = pca_results[100:130]
abaucin_pca = pca_results[130:]

# Plot the PCA results
plt.figure(figsize=(10, 8))
plt.scatter(training_pca[:, 0], training_pca[:, 1], color='blue', label='Training Dataset')
plt.scatter(prediction_pca[:, 0], prediction_pca[:, 1], color='red', label='Prediction Set')
plt.scatter(abaucin_pca[:, 0], abaucin_pca[:, 1], color='yellow', label='Abaucin', edgecolor='black', s=100)
plt.legend()
plt.title('PCA Plot Showing Chemical Relationships')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()