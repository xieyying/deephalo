import keras
from keras import layers
import tensorflow as tf

def model(input_shape,  output_shape):

    """自定义模型结构_单输出"""
    input = keras.Input(shape=(input_shape,), name="features1")
    # share = layers.Dense(32, activation="relu")(input)
    # share = layers.Dense(64, activation="relu")(share)
    input1 = input[:,:-2]
    input1 = layers.GaussianNoise(0.01)(input1)
   
    input2 = input[:,-2:]
    input2 = tf.pow(input2, 15)
    
    x = layers.Dense(64, activation="relu")(input1)
    y = layers.Dense(128, activation="relu")(input2)

    share = layers.concatenate([x, y])  
    share = layers.Dense(128, activation="relu")(share)
    share = layers.Dropout(.3)(share)
    # share = layers.Dense(512, activation="relu")(share)
    share = layers.Dense(128, activation="relu")(share)
    share = layers.Dense(32, activation="relu")(share)
    # share = layers.Dropout(.3)(share)
    clf_output = layers.Dense(output_shape,activation='softmax', name="classes")(share)
    clfs = keras.Model(inputs=input, outputs=clf_output, name="pick_halo_ann")

    return clfs

