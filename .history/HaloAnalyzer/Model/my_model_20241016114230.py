import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
import keras,os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from .methods import create_dataset
from .model_build import model
from keras import layers
import keras_tuner as kt
from functools import partial
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score, mean_absolute_percentage_error, mean_squared_error
from keras.utils import to_categorical



class MyModel:
    '''
    自定义模型类，包含数据集加载，模型构建，模型训练，模型评估等方法

    Functions:
    load_dataset: 加载数据集
    get_model: 获取模型
    train: 训练模型
    show_CM: 显示混淆矩阵
    work_flow: 模型训练流程
    '''

    def __init__(self,para) -> None:
        """
        初始化模型参数，为方便调用，参数以字典形式传入。

        Args:
        para: class, 模型参数, 包括batch_size, epochs, features, paths, test_path, classes, weight, learning_rate
        """
        self.para = para
        self.input_shape = len(self.para.features_list)

    def load_dataset(self):
        """
        加载数据集，创建训练集和验证集，以及测试集

        Returns:
        None
        """
        self.train_dataset,self.val_dataset,self.X_test, self.Y_test,self.val_ = create_dataset(self.para.features_list.copy(),self.para.paths,self.para.train_batch)
        
    def get_model(self):
        """
        加载预先定义的模型，并绘制模型结构图

        Returns:
        None
        """
        self.model = model(self.input_shape,self.para.classes)
        keras.utils.plot_model(self.model, to_file=r'./trained_models/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96)
    
    def train(self):
        """
        训练模型，并保存训练好的模型

        Returns:
        None
        """
        opt = tf.keras.optimizers.Adam(learning_rate=self.para.learning_rate)

        self.model.compile(optimizer=opt,
            loss={'classes': 'SparseCategoricalCrossentropy'},
                loss_weights={'classes': self.para.classes_weight},
            metrics=['accuracy'])

        # 创建EarlyStopping回调
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        self.history = self.model.fit(self.train_dataset, epochs=self.para.epochs, 
                                    validation_data=self.val_dataset, 
                                    callbacks=[early_stopping])  # 添加回调到fit函数
        self.model.summary()
        self.model.save(r'./trained_models/pick_halo_ann.h5') 
         
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
  

    def show_cm(self):
        """
        显示混淆矩阵，计算recall和precision，保存预测结果

        Returns:
        None
        """
        # load the model
        model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5')
        tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
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
            "ints_0",
            "ints_1",
            "ints_2",
            "ints_3",
            "ints_4",
            "ints_5",
            "m2_m1",
            "m1_m0",   
            ]

        formula_test = np.array(self.val_['formula'].tolist())
        m0 = np.array(self.val_['mz_0'].tolist())
        wrong_data = pd.DataFrame(wrong_data,columns=cols)
        wrong_data['formula'] = pd.Series(formula_test[wrong_index])
        wrong_data['mz_0'] = pd.Series(m0[wrong_index])
        wrong_data['true_classes'] = pd.Series(Y_val[wrong_index])
        wrong_data['pred_classes'] = pd.Series(y_pred[wrong_index])
        
        wrong_data.to_csv(r'./trained_models/pick_halo_ann_wrong_data.csv',index=False)
    
        # Compute confusion matrix
        confusion_matrix(Y_val, y_pred)

        # Compute recall and precision for each class
        report = classification_report(Y_val, y_pred, output_dict=True,zero_division=0)
        recalls = [report[str(i)]['recall'] if str(i) in report else 0  for i in range(8)]
        precisions = [report[str(i)]['precision'] if str(i) in report else 0 for i in range(8)]
        F1_sore = [report[str(i)]['f1-score'] if str(i) in report else 0 for i in range(8)]

        # Plot the confusion matrix
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot the confusion matrix
        ConfusionMatrixDisplay.from_predictions(Y_val, y_pred, ax=axs[0, 0], cmap=plt.cm.terrain)
        axs[0, 0].set_title('Classifier')
        
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
        plt.tight_layout()
        plt.show()
    
    def work_flow(self):
        """
        模型训练流程，包括加载数据集，获取模型，训练模型，显示混淆矩阵

        example:
        model = my_model(para)
        model.work_flow()
        """
        self.load_dataset()
        self.get_model()
        self.train()
        self.show_cm()
        
class MyHypermodel(kt.HyperModel):
    """
    自定义超参数搜索类，继承自kt.HyperModel

    Functions:
    __init__: 初始化参数
    build: 构建模型
    fit: 训练模型
    """
    def __init__(self,input_shape,output_shape,loss_weights):
        """
        初始化参数

        Args:
        input_shape: int, 输入特征的维度
        output_shape: int, 输出类别的数量
        loss_weights: dict, 损失权重
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.classes_weight = loss_weights
    
    def build(self, hp):
        """
        构建模型

        Args:
        hp: kt.HyperParameters, 超参数搜索空间

        Returns:
        model: keras.Model, 构建好的模型
        """
        # 定义batch_size的搜索空间
        hp.Choice('batch_size', [8,16,32,64,128],default=8)
        
        # input layer
        inputs = keras.Input(shape=(self.input_shape,), name="features1")
        
        input1 = inputs[:, :-2]
        #add noise
        input1 = layers.GaussianNoise(hp.Choice('noise1',[0.03]))(input1)
        input2 = inputs[:, -2:]
        input2 = layers.GaussianNoise(hp.Choice('noise2',[0.001]))(input2)
        #scaling
        power = hp.Choice('power_num', [0,5,10,15,20],default=0)
        input2 = tf.pow(input2,power)
        #wise feature
        x = layers.Dense(hp.Choice("units_0x", [16,32,64,128,256],default=16), activation="relu")(input1)
        y = layers.Dense(hp.Choice("units_0y", [8,16,32,64,128,256,512],default=8), activation="relu")(input2)
        share = layers.concatenate([x,y])
        # 定义神经网络的层数的搜索空间
        num_layers = hp.Int('num_layers', 1, 5)
        for i in range(num_layers):
            if i < num_layers:
                units = hp.Choice(f'units_{i+1}', [32,64,128,256,512],default=32)
                dropout = hp.Choice(f'dropout_{i+1}', [0.0,0.3],default=0)
                share = layers.Dense(units, activation="relu")(share)
                share = layers.Dropout(dropout)(share)
        # output layer
        clf_output = layers.Dense(self.output_shape, activation='softmax', name="classes")(share)
     
        model = keras.Model(inputs=inputs, outputs=clf_output, name="pick_halo_ann")
        learning_rate = hp.Choice('learning_rate', [0.0001,0.0003], default=0.0001)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                    loss_weights={'classes': self.classes_weight},)
        return model
        
    def fit(self,hp,model,X_train,Y_train,X_test,Y_test,**kwargs):
        """
        训练模型

        Args:
        hp: kt.HyperParameters, 超参数搜索空间
        model: keras.Model, 构建好的模型
        X_train: np.ndarray, 训练集特征
        Y_train: np.ndarray, 训练集标签
        X_test: np.ndarray, 验证集特征
        Y_test: np.ndarray, 验证集标签
        kwargs: dict, 其他参数

        Returns:
        history: dict, 训练历史
        """
        batch_size = hp.get('batch_size')
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
        val_dataset = val_dataset.shuffle(len(X_test)).batch(batch_size)
        history = model.fit(train_dataset, validation_data = val_dataset,**kwargs)
        return history
       
def my_search(para):
    """
    超参数搜索函数

    Args:
    para: class , 模型参数

    Returns:
    None
    """
    paths = para.paths
    features = para.features_list
    input_shape = len(features)
    output_shape = para.classes
    classes_weight = para.classes_weight
    
    X_train, Y_train, X_test, Y_test = create_dataset(features.copy(),paths,para.train_batch,model='search')
    hypermodel = MyHypermodel(input_shape,output_shape,classes_weight)
    tuner = kt.BayesianOptimization(hypermodel,
                    objective="val_accuracy",
                    max_trials=300, # 50-100
                    directory="noisy_mz_001_inty_03",
                    overwrite=False,
                    project_name="kt_base",
                    num_initial_points=5,)  # 2-10

    tuner.search(X_train,Y_train,X_test, Y_test, epochs=5, callbacks=[keras.callbacks.TensorBoard("./trained_models/tb_logs")],)
    tuner.results_summary()

if __name__ == '__main__':
    pass