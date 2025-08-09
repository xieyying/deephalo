import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

def prepare_training_dataset(features, paths, batch_size, model='base'):
    """
    Prepare the dataset required for model training in tf.data.Dataset format.

    Args:
    features: list, feature names.
    paths: list, paths to the dataset files.
    batch_size: int, batch size for training.
    model: str, type of model ('base' or 'search').

    Returns:
    train_dataset: tf.data.Dataset, training dataset.
    val_dataset: tf.data.Dataset, validation dataset.
    X_test: np.array, features of the validation set.
    Y_test: np.array, labels of the validation set.
    val_: pd.DataFrame, validation set as a DataFrame.
    """
    # Add 'group' to the list of features
    features += ['group']
    df = pd.DataFrame()

    # Load and concatenate datasets from the given paths
    for path in paths:
        df_ = pd.read_csv(path)
        df = pd.concat([df, df_], axis=0)
    
    # Filter data where 'mz_0' is less than or equal to 2000
    df = df[df['mz_0'] <= 2000]

    # Print the total number of samples
    print('Total_data: ', len(df))

    # Calculate and print the number of samples in each class
    class_counts = np.bincount(df['group'].values)
    print('Whole_data_class_counts: ', class_counts)

    # Split the data into training and validation sets
    train_, val_ = train_test_split(df, test_size=0.2, random_state=6)

    # Extract features and targets for training and validation sets
    train = train_[features]
    val = val_[features]
    train_target = train.pop('group')
    val_target = val.pop('group')

    X_train = train.values
    X_test = val.values

    Y_train = train_target.values
    Y_test = val_target.values

    # Create TensorFlow datasets for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
    val_dataset = val_dataset.shuffle(len(X_test)).batch(batch_size)
    
    # Return datasets and validation data based on the model type
    if model == 'base':
        return train_dataset, val_dataset, X_test, Y_test, val_
    elif model == 'search':
        return X_train, Y_train, X_test, Y_test    
    

