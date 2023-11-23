import keras
from keras import layers

def model(input_shape,  output_shape):
    """自定义模型结构_单输出"""
    input = keras.Input(shape=(input_shape,), name="features")
    share = layers.Dense(128, activation="relu")(input)
    share = layers.Dropout(0.3)(share)
    share = layers.Dense(64, activation="relu")(share)
    share = layers.Dense(32, activation="relu")(share)
    share = layers.Dense(16, activation="relu")(share)

    clf_output = layers.Dense(output_shape,activation='softmax', name="classes")(share)

    clfs = keras.Model(inputs=input, outputs=[clf_output], name="pick_halo_ann")

    return clfs

def model_sequence(input_shape,  output1_shape, output2_shape, output3_shape):
    """自定义模型结构_多输出"""
    input = keras.Input(shape=(input_shape,), name="features")
    share = layers.Dense(4000, activation="relu")(input)
    share = layers.Dropout(0.5)(share)
    share = layers.Dense(2000, activation="relu")(share)
    share = layers.Dropout(0.5)(share)
    share = layers.Dense(1000, activation="relu")(share)
    share = layers.Dropout(0.5,name='share')(share)

    x = layers.Dense(500, activation="relu")(share)
    clf_base_output = layers.Dense(output1_shape,activation='softmax', name="base")(x)

    
    y = layers.Dense(500, activation="relu")(share)
    clf_sub_output = layers.Dense(output2_shape,activation='softmax', name="sub")(y)

    z = layers.Dense(500, activation="relu")(share)
    clf_hydro_output = layers.Dense(output3_shape,activation='softmax', name="hydro")(z)

    clfs = keras.Model(inputs=input, outputs=[clf_base_output, clf_sub_output,clf_hydro_output], name="pick_halo_ann")

    return clfs