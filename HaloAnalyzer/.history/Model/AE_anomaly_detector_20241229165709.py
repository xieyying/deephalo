import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping

class AEData:
    def __init__(self, file_path, features, groups=[]):
        """
        Initialize AEData class.

        Args:
            file_path (str): Path to the data file.
            features (list): List of feature columns to be used.
            groups (list, optional): List of groups to filter the data. Defaults to [].
        """
        self.file_path = file_path
        self.features = features
        self.groups = groups
        self.data = self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """
        Load and preprocess data.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        data = pd.read_csv(self.file_path)
        if self.groups:
            data = data[data['group'].isin(self.groups)]
        data = data[self.features]
        return data

def load_trained_model(model_path, layer_name):
    """
    Load a trained model and extract the output of a specified layer.

    Args:
        model_path (str): Path to the trained model file.
        layer_name (str): Name of the layer to extract the output from.

    Returns:
        Model: A sub-model that outputs the specified layer's output.
    """
    # Load the trained model
    model = keras.models.load_model(model_path)
    model.summary()
    
    # Ensure the layer name is correct
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        print(f"Layer name '{layer_name}' not found. Please check the model's layer names.")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name}")
        raise
    
    # Create a sub-model
    units_1_output_model = Model(inputs=model.input, outputs=target_layer.output)
    return units_1_output_model

def load_and_preprocess_data(file_path, features, groups):
    """
    Load and preprocess data.

    Args:
        file_path (str): Path to the data file.
        features (list): List of feature columns to be used.
        groups (list): List of groups to filter the data.

    Returns:
        np.ndarray: Preprocessed training data.
    """
    X_train_df = AEData(file_path, features, groups=groups).data
    X_train = X_train_df[features].values
    return X_train

def build_autoencoder(input_dim):
    """
    Build an Autoencoder model.

    Args:
        input_dim (int): Input dimension.

    Returns:
        Model: Compiled Autoencoder model.
    """
    autoencoder_input = keras.Input(shape=(input_dim,), name="autoencoder_input")
    encoded = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(autoencoder_input)
    encoded = layers.Dense(16, activation='relu',kernel_initializer='he_normal')(encoded)
    decoded = layers.Dense(64, activation='relu',kernel_initializer='he_normal')(encoded)
    decoded = layers.Dense(input_dim)(decoded)
    
    autoencoder = Model(inputs=autoencoder_input, outputs=decoded, name="autoencoder")
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003)
    autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    return autoencoder

def train_autoencoder(autoencoder, data, epochs=50, batch_size=32):
    """
    Train the Autoencoder model.

    Args:
        autoencoder (Model): The Autoencoder model to be trained.
        data (np.ndarray): Training data.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        History: Training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping])
    return history

def recognize_level(model, X_train: pd.DataFrame, bins: int = 10000, color: str = 'blue') -> pd.DataFrame:
    """
    Predict the input data using the provided model, calculate the mean squared error (MSE) loss,
    and plot the distribution of the loss.

    Args:
        model: A trained Keras model used for prediction.
        X_train: A DataFrame containing the input data for prediction.
        bins: The number of bins to use for the histogram. Default is 1000.
        color: The color of the histogram. Default is 'blue'.

    Returns:
        pd.DataFrame: A DataFrame containing the loss (MSE) for each sample in X_train.
    """
    if X_train.empty:
        raise ValueError("X_train is empty.")

    # Predict the input data and calculate the loss
    X_pred = model.predict(np.array(X_train))
    X_pred = pd.DataFrame(X_pred, columns=X_train.columns, index=X_train.index)
    scored = pd.DataFrame(index=X_train.index)
    scored['Loss_mse'] = np.mean(np.power(X_pred - X_train, 2), axis=1)
    reconstruction_error = scored['Loss_mse']
    threshold_999 = np.quantile(reconstruction_error, 0.999)
    print(f"Anomaly detection threshold 99.9%: {threshold_999}")
    # Plot the distribution of the loss
    plt.figure()
    sns.histplot(scored['Loss_mse'], bins=bins, kde=False, color=color)
    plt.xlabel('Loss (MSE)')
    plt.ylabel('Number of samples')
    plt.title('Distribution of loss (MSE)')
    plt.show()
    return scored
    
def main():
    # Configure paths and parameters
    work_folder = 'D:\workissues\manuscript\halo_mining\HaloAnalyzer'
    model_path = os.path.join(work_folder, 'hyperparameter_optimization', 'noisy_and_peak_numbers', '0.001_inty_0.03_5_peaks', 'trained_models', 'pick_halo_ann.h5')
    layer_name = "dense"  # Adjust according to the model summary
    data_file = os.path.join(work_folder, 'datasets', 'training_validation_dataset', 'train_and_val', 'training_and_val_data_noise_added', 'inty_0.07_0.006_mz_0.0018', 'training_and_val_data.csv')
    
    features = [
        "p0_int",
        "p1_int",
        "p2_int",
        "p3_int",
        "p4_int",
        "m2_m1",
        "m1_m0", 
    ]
    groups = [0, 1, 2, 6]  # Adjust according to actual needs
    
    # Load the model and extract the specified layer's output
    units_1_output_model = load_trained_model(model_path, layer_name)
    
    # Load and preprocess data
    X_train = load_and_preprocess_data(data_file, features, groups)
    
    # Extract the specified layer's output
    units_1_output = units_1_output_model.predict(X_train)
    
    # Determine the input dimension of the Autoencoder
    input_dim = units_1_output.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Build the Autoencoder
    autoencoder = build_autoencoder(input_dim)
    
    # Train the Autoencoder
    history = train_autoencoder(autoencoder, units_1_output, epochs=10, batch_size=32)
    
    # Save the trained Autoencoder
    output = os.path.join(work_folder, 'anormal_detect', 'training_and_val_data_dense', '256_64_16_high_noise.h5')
    autoencoder.save(output)
    print("Autoencoder training completed and saved.")
    
    # Load the trained Autoencoder
    autoencoder = keras.models.load_model(output)
    print(autoencoder.summary())
    print('256_64_16_high_noise.h5')
    recognize_level(autoencoder, pd.DataFrame(units_1_output))
    
def anormal_isotope_pattern_detection(df_feature):
    """
    Detect anomalous isotope patterns in the given feature DataFrame.

    Args:
        df_feature (pd.DataFrame): DataFrame containing the feature data.

    Returns:
        pd.DataFrame: DataFrame with an additional column for reconstruction error.
    """
    querys_2 = df_feature[['p0_int', 'p1_int', 'p2_int', 'p3_int', 'p4_int']]
    querys_3 = df_feature[['m2_m1', 'm1_m0']]
    querys_2 = pd.concat([querys_2, querys_3], axis=1)
    querys_input = querys_2.values.astype('float64')
    
    work_folder = 'D:\workissues\manuscript\halo_mining\HaloAnalyzer'
    model_path = os.path.join(work_folder, 'hyperparameter_optimization', 'noisy_and_peak_numbers', '0.001_inty_0.03_5_peaks', 'trained_models', 'pick_halo_ann.h5')
    layer_name = 'dense'
    units_1_output_model = load_trained_model(model_path, layer_name)
    autoencoder_input = units_1_output_model.predict(querys_input)
    output = os.path.join(work_folder, 'anormal_detect', 'training_and_val_data_dense', '256_64_16_high_noise.h5')
    autoencoder_model = output
    anomy_sco = tf.keras.models.load_model(autoencoder_model)
    anomy_sco.summary()
    reconstructed_X = anomy_sco.predict(autoencoder_input)
    
    reconstruction_error = np.mean(np.power(autoencoder_input - reconstructed_X, 2), axis=1)
    df_feature.loc[:, 'reconstruction_error'] = reconstruction_error
    return df_feature

def plot_quantile_curve(df, col='reconstruction_error'):
    """
    Plot the quantile curve and find the point with the maximum slope.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name to calculate quantiles. Default is 'reconstruction_error'.
    """
    thresholds = np.linspace(0, 1, 1000)
    quantiles = [np.quantile(df[col], t) for t in thresholds]

    # Calculate the slope of the curve
    slopes = np.diff(quantiles) / np.diff(thresholds)
    max_slope_index = np.argmax(slopes)
    max_slope_threshold = thresholds[max_slope_index]
    max_slope_quantile = quantiles[max_slope_index]
    print(f'Max slope point: Threshold = {max_slope_threshold}, Quantile = {max_slope_quantile}')

    # Plot the quantile curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, quantiles, label='Quantile Curve')
    plt.scatter(max_slope_threshold, max_slope_quantile, color='red', label='Max Slope Point')
    plt.xlabel('Threshold')
    plt.ylabel(f'Quantile of {col}')
    plt.title(f'Quantile Curve of {col}')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f'Max slope point: Threshold = {max_slope_threshold}, Quantile = {max_slope_quantile}')
   
if __name__ == "__main__":
    # main()
    
    input = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\train_and_val\training_and_val_data_noise_added\inty_0.07_0.006_mz_0.0018\training_and_val_data.csv'
    df_feature = pd.read_csv (input)
    #group = 0, 1, 2, 6
    df_feature = df_feature[df_feature['group'].isin([0,1,2,6])]   
    df_ = anormal_isotope_pattern_detection(df_feature)
    df_.to_csv(input.replace('.csv', '_reconstruction_error.csv'), index=False)
    plot_quantile_curve(df_, col='reconstruction_error')
    
  
 