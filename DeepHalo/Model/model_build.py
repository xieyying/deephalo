import keras
from keras import layers
import tensorflow as tf

def model(input_shape,  output_shape):

    """自定义模型结构_单输出"""
    input = keras.Input(shape=(input_shape,), name="features1")
    
    input1 = input[:,:-2]
    input1 = layers.GaussianNoise(0.03)(input1)
   
    input2 = input[:,-2:]
    input2 = layers.GaussianNoise(0.001)(input2)
    input2 = tf.pow(input2, 20)
    
    x = layers.Dense(256, activation="relu")(input1)
    y = layers.Dense(512, activation="relu")(input2)

    share = layers.concatenate([input1, input2])  
    share = layers.Dense(256, activation="relu")(share) # layer1
    share = layers.Dropout(.3)(share)    #layer1
    # share = layers.Dense(512, activation="relu")(share) # layer2
    # share = layers.Dropout(.3)(share)    #layer2
    # share = layers.Dense(512, activation="relu")(share) # layer3
    # share = layers.Dropout(.3)(share)    #layer3
    # share = layers.Dense(32, activation="relu")(share) # layer4
    # # share = layers.Dropout(.3)(share)    #layer4
    # share = layers.Dense(32, activation="relu")(share) # layer5
    # # share = layers.Dropout(.3)(share)    #layer5
    clf_output = layers.Dense(output_shape,activation='softmax', name="classes")(share)
    clfs = keras.Model(inputs=input, outputs=clf_output, name="pick_halo_ann")

    return clfs

if __name__ == "__main__":
    model =keras.models.load_model(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\200_trails_mz_noise_0.001_inty_0.03_5_peaks\trained_models\pick_halo_ann.h5')
    model.summary()

