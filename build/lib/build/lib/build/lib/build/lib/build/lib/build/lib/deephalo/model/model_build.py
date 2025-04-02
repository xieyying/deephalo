import keras
from keras import layers
import tensorflow as tf

def isonn_model(input_shape, output_shape):
    """
    Isotope neural network model for uncommon element classification tasks.
    This model employs a dual-branch architecture:
    - One branch processes the intensity of isotope peaks.
    - The other branch processes the mass differences of isotopes.
    The outputs from both branches are combined to classify the input features into different groups based on the types of elements.

    Args:
    input_shape: int, the shape of the input features.
    output_shape: int, the number of output classes.

    Returns:
    clfs: keras.Model, the constructed model.
    """
    # Input layer
    input = keras.Input(shape=(input_shape,), name="features1")
    
    # Process the first branch (intensity of isotope peaks)
    input1 = input[:, :-2]
    input1 = layers.GaussianNoise(0.03)(input1)
    input1 = layers.Dense(256, activation="relu")(input1)
   
    # Process the second branch (mass differences of isotopes)
    input2 = input[:, -2:]
    input2 = layers.GaussianNoise(0.001)(input2)
    input2 = tf.pow(input2, 20)
    input2 = layers.Dense(512, activation="relu")(input2)

    # Combine the processed inputs
    share = layers.concatenate([input1, input2])  
    share = layers.Dense(256, activation="relu")(share)  # First shared dense layer
    share = layers.Dropout(0.3)(share)  # Dropout layer

    # Output layer
    clf_output = layers.Dense(output_shape, activation='softmax', name="classes")(share)

    # Build the model
    clfs = keras.Model(inputs=input, outputs=clf_output, name="pick_halo_ann")

    return clfs