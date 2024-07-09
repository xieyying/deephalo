import keras
from keras import layers
import tensorflow as tf

def model(input_shape,  output_shape):

    """自定义模型结构_单输出"""
    input = keras.Input(shape=(input_shape,), name="features1")
    # share = layers.Dense(32, activation="relu")(input)
    # share = layers.Dense(64, activation="relu")(share)
    input1 = input[:,:-2]
    input1 = layers.GaussianNoise(0.02)(input1)
   
    input2 = input[:,-2:]
    # input2 =layers.GaussianNoise(0.001)(input2)
    input2 = tf.pow(input2, 15)
    
    x = layers.Dense(256, activation="relu")(input1) # units_0x
    y = layers.Dense(8, activation="relu")(input2)  # units_0y

    share = layers.concatenate([x, y])  
    share = layers.Dense(32, activation="relu")(share) # units_1
    share = layers.Dropout(.3)(share)  # dropout_1
    share = layers.Dense(512, activation="relu")(share) # units_2
    # share = layers.Dropout(.3)(share) # dropout_2
    share = layers.Dense(32, activation="relu")(share) # units_3
    share = layers.Dropout(.3)(share) # dropout_3
    share = layers.Dense(512, activation="relu")(share) # units_4
    share = layers.Dropout(.3)(share) # dropout_4
    share = layers.Dense(32, activation="relu")(share) # units_5
    # share = layers.Dropout(.3)(share) # dropout_5
   
    clf_output = layers.Dense(output_shape,activation='softmax', name="classes")(share)
    clfs = keras.Model(inputs=input, outputs=clf_output, name="pick_halo_ann")

    return clfs

