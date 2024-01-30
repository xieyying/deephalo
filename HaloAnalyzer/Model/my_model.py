import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
import keras,os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from .methods import create_dataset,create_dataset_test,create_dataset_copy
from .model_build import model,copy_model,con_model,model_noise,transfer_model
from keras import layers
import keras_tuner as kt
from functools import partial
from sklearn.inspection import permutation_importance
import pickle
class my_model:
    '''自定义模型类，包含数据集加载，模型构建，模型训练，模型评估等方法'''
    def __init__(self,para) -> None:
        """para是一个字典，包含了模型训练所需的所有参数"""
        self.batch_size = para['batch_size']
        self.epochs = para['epochs']
        self.features = para['features']
        self.paths = para['paths']
        self.test_path = para['test_path']
        self.input_shape = len(self.features)
        self.output_shape = para['classes']
        self.classes_weight = para['weight']
        self.learning_rate = para['learning_rate']

    def load_dataset(self):
        """加载数据集"""
        self.train_dataset,self.val_dataset,self.X_test, self.Y_test,self.val_ = create_dataset(self.features.copy(),self.paths,self.batch_size)
       
        self.test_dataset,self.test_df = create_dataset_test(self.features.copy(),self.test_path,self.batch_size)
        # 计算每个类别的样本数量
        class_counts = np.bincount(self.Y_test)

        # 计算类别权重
        # self.class_weights = 1 / class_counts

        # 计算root CSW权重
        self.root_csw_weights = 1 / np.sqrt(class_counts)
        print(self.root_csw_weights)

        # 计算square CSW权重
        self.square_csw_weights = 1 / (class_counts ** 2)
        # print(self.square_csw_weights)

        #用tf

        # 计算样本权重
        # molecular_weights = np.array([sample[0][-1] for sample in self.train_dataset])
        # #
        # self.sample_weight = np.where(molecular_weights > 2000, 0.01, 1)
     
        # 在训练集合严重集中去掉m0_mz
        # self.train_dataset = self.train_dataset.map(lambda x, y: (x[:, :-1], y))
        # self.val_dataset = self.val_dataset.map(lambda x, y: (x[:, :-1], y))
        
    def get_model(self):
        """获取自定义模型，并绘制模型结构图"""
        #model_build中可以定义多种模型结构方便切换
        # self.model = model_noise(self.input_shape,self.output_shape)
        self.model = model_noise(self.input_shape,self.output_shape)
        #绘制模型图
        keras.utils.plot_model(self.model, to_file=r'./trained_models/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96)
    
    def train(self):
        """训练模型"""
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # 定义一个使用CSW权重的损失函数
        def csw_loss(y_true, y_pred):
            weights = tf.gather(self.class_weights, tf.cast(y_true, tf.int32))
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)

        # 定义一个使用root CSW权重的损失函数
        def root_csw_loss(y_true, y_pred):
            weights = tf.gather(self.root_csw_weights, tf.cast(y_true, tf.int32))
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)

        # 定义一个使用square CSW权重的损失函数
        def square_csw_loss(y_true, y_pred):
            weights = tf.gather(self.square_csw_weights, tf.cast(y_true, tf.int32))
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)
        
        self.model.compile(optimizer=opt,
            loss={'classes': 'SparseCategoricalCrossentropy'},
            # loss={'classes': root_csw_loss},
                loss_weights={'classes': self.classes_weight},
            metrics=['accuracy'])

        # 创建EarlyStopping回调
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        self.history = self.model.fit(self.train_dataset, epochs=self.epochs, 
                                    validation_data=self.val_dataset, 
                                    callbacks=[early_stopping])  # 添加回调到fit函数
        self.model.summary()

        # 显示loss和epoch的关系
        history = self.history.history
        print(history.keys())
        #不同loss显示不同颜色
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(history['loss'],color='red')
        plt.plot(history['accuracy'],color='blue')
        plt.plot(history['val_loss'],color='black')
        plt.plot(history['val_accuracy'],color='green')

        #显示图例
        plt.legend(['loss','accuracy','val_loss','val_accuracy'], loc='right')
        self.model.save(r'./trained_models/pick_halo_ann.h5')
        # 计算 Permutation Feature Importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        # 保存结果
        with open('./importance_result.pkl', 'wb') as f:
            pickle.dump(result, f)

        # 打印特征重要性
        for i in range(X.shape[1]):
            print(f"Feature {i}: {result.importances_mean[i]}")

    def train_transfer(self):
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        base_model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5')
        base_model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer('dense_2').output)
        base_model.trainable = False
        self.trans_model = transfer_model(self.input_shape,self.output_shape,base_model)

        #Train the top layer
        self.trans_model.compile(optimizer=opt,
                                 loss={'classes': 'SparseCategoricalCrossentropy'},
                                    loss_weights={'classes': self.classes_weight},
                                    metrics=['accuracy'])
        self.trans_model.summary()

        self.trans_model.fit(self.train_dataset, epochs=5, validation_data=self.val_dataset)
        
        #Do a round of fine-tuning of the entire model
        base_model.trainable = True
        self.trans_model.summary()
        self.trans_model.compile(optimizer=keras.optimizers.Adam(1e-5),
                                 loss={'classes': 'SparseCategoricalCrossentropy'},
                                    metrics=['accuracy'])
        self.trans_model.fit(self.train_dataset, epochs=2, validation_data=self.val_dataset, sample_weight=self.sample_weight)
        self.trans_model.save(r'./trained_models/pick_halo_transfer.h5')


    def show_CM(self):
        """显示混淆矩阵"""

        # load the model
        model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5')
        def csw_loss(y_true, y_pred):
            weights = tf.gather(self.class_weights, tf.cast(y_true, tf.int32))
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)
        # 定义一个使用root CSW权重的损失函数
        def root_csw_loss(y_true, y_pred):
            weights = tf.gather(self.root_csw_weights, tf.cast(y_true, tf.int32))
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)
        # 定义一个使用square CSW权重的损失函数
        def square_csw_loss(y_true, y_pred):
            weights = tf.gather(self.square_csw_weights, tf.cast(y_true, tf.int32))
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)
        # model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5', custom_objects={'root_csw_loss': root_csw_loss})

        # rankdir='LR' is used to make the graph horizontal.
        tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
        # make predictions on the validation set
       
        # X_val = self.X_test[:, :-1]

        X_val = self.X_test

        Y_val = self.Y_test
        y_pred = model.predict(X_val)

        # Convert y_pred to a DataFrame
        df_pred = pd.DataFrame(y_pred)

        # Add Y_val as a new column
        df_pred['Y_val'] = Y_val

        # Save to csv file
        df_pred.to_csv(r'./trained_models/prediction.csv',index=False)

        y_pred = np.argmax(y_pred, axis=1)
        wrong_index = np.where(Y_val !=y_pred)
    
        wrong_data = X_val[wrong_index]

        cols  = [
                # "ints_b_3",
                "ints_b_2",
                "ints_b_1",
                "ints_b0",
                "ints_b1",
                "ints_b2",
                "ints_b3",
                "m2_m1",
                "m1_m0",
                # 'b2_b1',
            ]

        formula_test = np.array(self.val_['formula'].tolist())
        m0 = np.array(self.val_['m0_mz'].tolist())
        wrong_data = pd.DataFrame(wrong_data,columns=cols)
        wrong_data['formula'] = pd.Series(formula_test[wrong_index])
        wrong_data['m0_mz'] = pd.Series(m0[wrong_index])
        wrong_data['true_classes'] = pd.Series(Y_val[wrong_index])
        wrong_data['pred_classes'] = pd.Series(y_pred[wrong_index])
        
        #将wrong_data转为dataframe
        wrong_data.to_csv(r'./trained_models/pick_halo_ann_wrong_data.csv',index=False)
    
        # # Compute confusion matrix
        cm = confusion_matrix(Y_val, y_pred)

        # Compute recall and precision for each class
        report = classification_report(Y_val, y_pred, output_dict=True,zero_division=0)
        # Compute recall and precision for each class
        report = classification_report(Y_val, y_pred, output_dict=True,zero_division=1)
        recalls = [report[str(i)]['recall'] if str(i) in report else 0 for i in range(cm.shape[0])]
        precisions = [report[str(i)]['precision'] if str(i) in report else 0 for i in range(cm.shape[0])]
        F1_sore = [report[str(i)]['f1-score'] if str(i) in report else 0 for i in range(cm.shape[0])]

        # Plot the confusion matrix
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot the confusion matrix
        ConfusionMatrixDisplay.from_predictions(Y_val, y_pred, ax=axs[0, 0], cmap=plt.cm.terrain)
        axs[0, 0].set_title('Classifier')
        # Plot the precision bar chart below the confusion matrix
        # Plot the precision bar chart below the confusion matrix
        colors = np.random.rand(len(precisions), 3)
        axs[1, 0].bar(np.arange(len(precisions)), precisions, color=colors)
        axs[1, 0].set_title('Precision')
        axs[1, 0].set_xlabel('Class')
        axs[1, 0].set_ylabel('Precision')
        #将数值标注在图上
        for i, v in enumerate(precisions):
            axs[1, 0].text(i - 0.25, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')

        # Plot the recall bar chart to the right of the confusion matrix
        
        axs[0, 1].barh(np.arange(len(recalls)), recalls, color=colors)
        axs[0, 1].set_title('Recall')
        axs[0, 1].set_xlabel('Recall')
        axs[0, 1].set_ylabel('Class')
        #将数值标注在图上
        for i, v in enumerate(recalls):
            axs[0, 1].text(v + 0.01, i + .25, str(round(v, 3)), color='black', fontweight='bold')
        axs[0, 1].invert_yaxis()  # Reverse the y-axis so class 0 is on top
        axs[0, 1].set_xlim(0, 1)  # Set the x-axis range to 0-1

        # Plot the f1-score bar chart below the recall bar chart
        axs[1, 1].bar(np.arange(len(F1_sore)), F1_sore, color=colors)
        axs[1, 1].set_title('F1-score')
        axs[1, 1].set_xlabel('Class')
        axs[1, 1].set_ylabel('F1-score')
        #将数值标注在图上
        for i, v in enumerate(F1_sore):
            axs[1, 1].text(i - 0.25, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')
        axs[1, 1].set_ylim(0, 1)
        # axs[1, 1].set_xlim(0, 1)

        plt.tight_layout()
        plt.show()
    
    def work_flow(self):
        """模型训练流程"""
        if not os.path.exists('./trained_models'):
            os.mkdir('./trained_models')
        self.load_dataset()
        self.get_model()
        self.train()
        # self.train_transfer()
        self.show_CM()

def show_CM(model_path):
    """显示混淆矩阵"""

    # load the model
    model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5')
    def csw_loss(y_true, y_pred):
        weights = tf.gather(self.class_weights, tf.cast(y_true, tf.int32))
        weights = tf.cast(weights, tf.float32)
        return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)
    # 定义一个使用root CSW权重的损失函数
    def root_csw_loss(y_true, y_pred):
        weights = tf.gather(self.root_csw_weights, tf.cast(y_true, tf.int32))
        weights = tf.cast(weights, tf.float32)
        return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)
    # 定义一个使用square CSW权重的损失函数
    def square_csw_loss(y_true, y_pred):
        weights = tf.gather(self.square_csw_weights, tf.cast(y_true, tf.int32))
        weights = tf.cast(weights, tf.float32)
        return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights)
    # model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5', custom_objects={'root_csw_loss': root_csw_loss})

    # rankdir='LR' is used to make the graph horizontal.
    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    # make predictions on the validation set
    
    # X_val = self.X_test[:, :-1]

    X_val = self.X_test

    Y_val = self.Y_test
    y_pred = model.predict(X_val)

    # Convert y_pred to a DataFrame
    df_pred = pd.DataFrame(y_pred)

    # Add Y_val as a new column
    df_pred['Y_val'] = Y_val

    # Save to csv file
    df_pred.to_csv(r'./trained_models/prediction.csv',index=False)

    y_pred = np.argmax(y_pred, axis=1)
    wrong_index = np.where(Y_val !=y_pred)

    wrong_data = X_val[wrong_index]

    cols  = [
            # "ints_b_3",
            "ints_b_2",
            "ints_b_1",
            "ints_b0",
            "ints_b1",
            "ints_b2",
            "ints_b3",
            "m2_m1",
            "m1_m0",
            # 'b2_b1',
        ]

    formula_test = np.array(self.val_['formula'].tolist())
    m0 = np.array(self.val_['m0_mz'].tolist())
    wrong_data = pd.DataFrame(wrong_data,columns=cols)
    wrong_data['formula'] = pd.Series(formula_test[wrong_index])
    wrong_data['m0_mz'] = pd.Series(m0[wrong_index])
    wrong_data['true_classes'] = pd.Series(Y_val[wrong_index])
    wrong_data['pred_classes'] = pd.Series(y_pred[wrong_index])
    
    #将wrong_data转为dataframe
    wrong_data.to_csv(r'./trained_models/pick_halo_ann_wrong_data.csv',index=False)

    # # Compute confusion matrix
    cm = confusion_matrix(Y_val, y_pred)

    # Compute recall and precision for each class
    report = classification_report(Y_val, y_pred, output_dict=True,zero_division=0)
    # Compute recall and precision for each class
    report = classification_report(Y_val, y_pred, output_dict=True,zero_division=1)
    recalls = [report[str(i)]['recall'] if str(i) in report else 0 for i in range(cm.shape[0])]
    precisions = [report[str(i)]['precision'] if str(i) in report else 0 for i in range(cm.shape[0])]
    F1_sore = [report[str(i)]['f1-score'] if str(i) in report else 0 for i in range(cm.shape[0])]

    # Plot the confusion matrix
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot the confusion matrix
    ConfusionMatrixDisplay.from_predictions(Y_val, y_pred, ax=axs[0, 0], cmap=plt.cm.terrain)
    axs[0, 0].set_title('Classifier')
    # Plot the precision bar chart below the confusion matrix
    # Plot the precision bar chart below the confusion matrix
    colors = np.random.rand(len(precisions), 3)
    axs[1, 0].bar(np.arange(len(precisions)), precisions, color=colors)
    axs[1, 0].set_title('Precision')
    axs[1, 0].set_xlabel('Class')
    axs[1, 0].set_ylabel('Precision')
    #将数值标注在图上
    for i, v in enumerate(precisions):
        axs[1, 0].text(i - 0.25, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')

    # Plot the recall bar chart to the right of the confusion matrix
    
    axs[0, 1].barh(np.arange(len(recalls)), recalls, color=colors)
    axs[0, 1].set_title('Recall')
    axs[0, 1].set_xlabel('Recall')
    axs[0, 1].set_ylabel('Class')
    #将数值标注在图上
    for i, v in enumerate(recalls):
        axs[0, 1].text(v + 0.01, i + .25, str(round(v, 3)), color='black', fontweight='bold')
    axs[0, 1].invert_yaxis()  # Reverse the y-axis so class 0 is on top
    axs[0, 1].set_xlim(0, 1)  # Set the x-axis range to 0-1

    # Plot the f1-score bar chart below the recall bar chart
    axs[1, 1].bar(np.arange(len(F1_sore)), F1_sore, color=colors)
    axs[1, 1].set_title('F1-score')
    axs[1, 1].set_xlabel('Class')
    axs[1, 1].set_ylabel('F1-score')
    #将数值标注在图上
    for i, v in enumerate(F1_sore):
        axs[1, 1].text(i - 0.25, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')
    axs[1, 1].set_ylim(0, 1)
    # axs[1, 1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()



class Myhypermodel(kt.HyperModel):
    def __init__(self,input_shape,output_shape,loss_weights):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.classes_weight = loss_weights
    
    def get_config(self,hp):
        batch_size = hp.Int('batch_size', min_value=32, max_value=512, step=32)
        return {'batch_size':batch_size}
    # def build1(self, hp):
        
    #     inputs = keras.Input(shape=(self.input_shape,), name="features1")
    #     print('input shape',inputs.shape)
    #     input1 = inputs[:, :-3]
    #     input1 = layers.GaussianNoise(hp.Choice('noise1',[0.02]))(input1)
    #     input2 = inputs[:, -3:]
    #     input2 = layers.GaussianNoise(hp.Choice('noise2',[0.002]))(input2)
    #     input2 = layers.Lambda(lambda x: x ** 5 - 0.1)(input2)

    #     x = layers.Dense(hp.Choice("units_0x", [32,64,128,256,512],default=32), activation="relu")(input1)
    #     y = layers.Dense(hp.Choice("units_0y", [32,64,128,256,512],default=32), activation="relu")(input2)

    #     share = layers.concatenate([x,y])

    #     for i in range(hp.Int('num_layers', 1, 5)):
    #         share = layers.Dense(hp.Choice(f'units_{i}', [32,64,128,256,512],default=32), activation="relu")(share)
    #         share = layers.Dropout(hp.Choice(f'dropout_{i}', [0.0,0.2,0.3],default=0))(share)

    #     clf_output = layers.Dense(self.output_shape, activation='softmax', name="classes")(share)

    #     model = keras.Model(inputs=inputs, outputs=clf_output, name="pick_halo_ann")
        
    #     optimizer_choice = hp.Choice('optimizer', ['adam', 'adadelta'])
    #     learning_rate = hp.Choice('learning_rate', [0.0003,0.0001], default=0.0003)

    #     if optimizer_choice == 'adam':
    #         optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    #     elif optimizer_choice == 'adadelta':
    #         optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate)

    #     model.compile(optimizer=optimizer,
    #                 loss="sparse_categorical_crossentropy",
    #                 metrics=["accuracy"],
    #                 loss_weights={'classes': self.classes_weight},)
    #     return model

    def build(self, hp):
        
        inputs = keras.Input(shape=(self.input_shape,), name="features1")

        input1 = inputs[:, :-3]
        input1 = layers.GaussianNoise(hp.Choice('noise1',[0.01,0.02,0.03]))(input1)
        input2 = inputs[:, -3:]
        input2 = layers.GaussianNoise(hp.Choice('noise2',[0.002,0.003]))(input2)
        input2 = layers.Lambda(lambda x: x ** 5 - 0.1)(input2)

        x = layers.Dense(hp.Choice("units_0x", [32,64,128,256,512],default=32), activation="relu")(input1)
        y = layers.Dense(hp.Choice("units_0y", [32,64,128,256,512],default=32), activation="relu")(input2)

        share = layers.concatenate([x,y])

        # share = layers.Dense(hp.Choice(f'units_{0}', [32,64,128,256,512],default=32), activation="relu")(inputs)
        # share = layers.Dropout(hp.Choice(f'dropout_{0}', [0.0,0.2,0.3],default=0))(inputs)
        
        for i in range(hp.Int('num_layers', 1, 5)):
            share = layers.Dense(hp.Choice(f'units_{i+1}', [32,64,128,256,512],default=32), activation="relu")(share)
            share = layers.Dropout(hp.Choice(f'dropout_{i+1}', [0.0,0.2,0.3],default=0))(share)

        clf_output = layers.Dense(self.output_shape, activation='softmax', name="classes")(share)

        model = keras.Model(inputs=inputs, outputs=clf_output, name="pick_halo_ann")
        
        # optimizer_choice = hp.Choice('optimizer', ['adam', 'adadelta'])
        learning_rate = hp.Choice('learning_rate', [0.0003,0.0001], default=0.0003)

        # if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # elif optimizer_choice == 'adadelta':
        #     optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                    loss_weights={'classes': self.classes_weight},)
        return model
    
    def fit(self,hp,model,X_train,Y_train,X_test,Y_test,**kwargs):
        batch_size = hp.Choice('batch_size', [32,64,128,256],default=32)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
        val_dataset = val_dataset.shuffle(len(X_test)).batch(batch_size)
        return model.fit(train_dataset, validation_data = val_dataset,**kwargs)
       
def my_search(para):
    # batch_size = para['batch_size']
    paths = para['paths']
    features = para['features']
    test_path = para['test_path']
    input_shape = len(features)
    output_shape = para['classes']
    classes_weight = para['weight']

    X_train, Y_train, X_test, Y_test = create_dataset_copy(features.copy(),paths)

    hp = kt.HyperParameters()
    hypermodel = Myhypermodel(input_shape,output_shape,classes_weight)

    tuner = kt.BayesianOptimization(hypermodel,
                    objective="val_accuracy",
                    #  max_epochs=10,
                    #  factor=3,
                    max_trials=90,
                    directory="my_search_9_feature",
                    overwrite=True,
                    project_name="kt_base",
                    num_initial_points=4,)
                    #  hyperban_iterations=1,)
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # tuner.search(X_train,Y_train,X_test, Y_test, epochs=10, callbacks=[stop_early])
    # log_dir = "logs/fit/" + time.strftime("run_%Y_%m_%d-%H_%M_%S")
    # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # tuner.search(X_train,Y_train,X_test, Y_test, epochs=5, callbacks=[keras.callbacks.TensorBoard(log_dir)])
    tuner.search(X_train,Y_train,X_test, Y_test, epochs=5, callbacks=[keras.callbacks.TensorBoard("./trained_models/tb_logs")],)
    best_hps = tuner.get_best_hyperparameters(num_trials=2)[0]
    tuner.results_summary()
    batch_size = best_hps.get('batch_size')
    # epoch_num = best_hps.get('epoch_num')

    print(best_hps.values)
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test),batch_size=batch_size)
    model.summary()
    # model.save(r'./trained_models/pick_halo_ann_search.h5')
    # model.evaluate(test_dataset)


if __name__ == '__main__':
    pass