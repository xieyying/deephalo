import keras
from keras import layers
import tensorflow as tf

def model(input_shape,  output_shape):
    def my_fun2(x):
        data = x[:,:-2]
        last_column = x[:,-1:]
        second_last_column = x[:,-2:-1]
        new_last_column = tf.pow(last_column, 10)
        new_second_last_column = tf.pow(second_last_column, 10)
        new_x = tf.concat([data, new_last_column], axis=1)
        new_x = tf.concat([new_x, new_second_last_column], axis=1)
        return new_x
    def my_fun1(x):
        data = x[:,:-1]
        last_column = x[:,-1:]
        new_last_column = tf.pow(last_column, 10)
        new_x = tf.concat([data, new_last_column], axis=1)
        return new_x

    """自定义模型结构_单输出"""
    input = keras.Input(shape=(input_shape,), name="features")
    new_input = layers.Lambda(my_fun1)(input)
    share = layers.Dense(256, activation="relu")(new_input)
    share = layers.Dropout(0.3)(share)
    share = layers.Dense(128, activation="relu")(new_input)
    # share = layers.Dropout(0.3)(share)
    share = layers.Dense(64, activation="relu")(share)
    share = layers.Dense(32, activation="relu")(share)
    share = layers.Dense(16, activation="relu")(share)

    clf_output = layers.Dense(output_shape,activation='softmax', name="classes")(share)

    clfs = keras.Model(inputs=input, outputs=[clf_output], name="pick_halo_ann")

    return clfs

def copy_model(input_shape,  output_shape):


    """自定义模型结构_单输出"""
    input_ = keras.Input(shape=(input_shape,), name="features")
    share = layers.Dense(512, activation="relu")(input_)
    share = layers.Dropout(0.5)(share) 
    share = layers.Dense(256, activation="relu")(share)
    share = layers.Dropout(0.2)(share)    
    share = layers.Dense(128, activation="relu")(share)
    share = layers.Dropout(0.1)(share)
    share = layers.Dense(64, activation="relu")(share)
    share = layers.Dense(32, activation="relu")(share)
    share = layers.Dense(16, activation="relu")(share)

    clf_output = layers.Dense(output_shape,activation='softmax', name="classes")(share)

    clfs = keras.Model(inputs=input_, outputs=[clf_output], name="pick_halo_ann")

    return clfs

def con_model(input_shape,  output_shape):


    """自定义模型结构_单输出"""
    input = keras.Input(shape=(input_shape,), name="features1")
    input1 = input[:,:-3]
    input2 = input[:,-3:]
    x = layers.Dense(128, activation="relu")(input1)

    y = layers.Dense(128, activation="relu")(input2)
    share = layers.concatenate([x, y])  
    share = layers.Dense(128, activation="relu")(share)
    share = layers.Dropout(.2)(share)
    share = layers.Dense(64, activation="relu")(share)
    share = layers.Dense(32, activation="relu")(share)
    share = layers.Dense(16, activation="relu")(share)

    clf_output = layers.Dense(output_shape,activation='softmax', name="classes")(share)

    clfs = keras.Model(inputs=input, outputs=clf_output, name="pick_halo_ann")

    return clfs




if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow import keras
    def my_fun(x):
        # 分割数据
        data, last_column = tf.split(x, [x.shape[1] - 1, 1], axis=1)
        data = tf.cast(data,tf.float32)

        # 计算最后一列的10次方
        new_last_column = tf.pow(last_column, 10)
        new_last_column = tf.cast(new_last_column,tf.float32)

        # 合并新的列与原始数据
        new_x = tf.concat([data, new_last_column], axis=1)

        return new_x

    # x = tf.random.uniform([10000, 9], minval=1, maxval=10, dtype=tf.int32)
    # new_x = layers.Lambda(my_fun)(x)

    x = tf.constant([2, 3, 4, 5, 6, 7, 8, 9, 1.003])
    x = tf.reshape(x, [3, 3])
    new_x = layers.Lambda(my_fun)(x)
    print(new_x),print(x)
    print(4**10,7**10,1.003**10)