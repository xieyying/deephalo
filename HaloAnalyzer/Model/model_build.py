import keras
from keras import layers

def model(input_shape,  output1_shape, output2_shape, output3_shape):
    input = keras.Input(shape=(input_shape,), name="img")
    share = layers.Dense(4000, activation="relu")(input)
    share = layers.Dense(2000, activation="relu")(share)
    share = layers.Dropout(0.5)(share)
    share = layers.Dense(1000, activation="relu")(share)
    share = layers.Dropout(0.5,name='share')(share)

    x = layers.Dense(500, activation="relu")(share)
    clf_base_output = layers.Dense(output1_shape,activation='softmax', name="base")(x)

    y = layers.Concatenate()([share, clf_base_output,input])
    y = layers.Dense(500, activation="relu")(y)
    clf_sub_output = layers.Dense(output2_shape,activation='softmax', name="sub")(y)

    z = layers.Concatenate()([share, clf_base_output,clf_sub_output,input])
    z = layers.Dense(500, activation="relu")(z)
    clf_hydro_output = layers.Dense(output3_shape,activation='softmax', name="hydro")(z)

    clfs = keras.Model(inputs=input, outputs=[clf_base_output, clf_sub_output,clf_hydro_output], name="pick_halo_ann")

    return clfs