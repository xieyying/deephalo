import pandas as pd
import tensorflow as tf
import keras,os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .methods import create_dataset
from .model_build import model,copy_model,con_model,model_noise,transfer_model
from keras import layers
class my_model:
    '''自定义模型类，包含数据集加载，模型构建，模型训练，模型评估等方法'''
    def __init__(self,para) -> None:
        """para是一个字典，包含了模型训练所需的所有参数"""
        self.batch_size = para['batch_size']
        self.epochs = para['epochs']
        self.features = para['features']
        self.paths = para['paths']
        self.input_shape = len(self.features)
        self.output_shape = para['classes']
        self.classes_weight = para['weight']
        self.learning_rate = para['learning_rate']
    def load_dataset(self):
        """加载数据集"""
        self.train_dataset,self.val_dataset,self.X_test, self.Y_test,self.val_ = create_dataset(self.features,self.paths,self.batch_size)

    def get_model(self):
        """获取自定义模型，并绘制模型结构图"""
        #model_build中可以定义多种模型结构方便切换
        self.model = model_noise(self.input_shape,self.output_shape)
        #绘制模型图
        keras.utils.plot_model(self.model, to_file=r'./trained_models/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96)
    
    def train(self):
        """训练模型"""
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=opt,
              loss={'classes': 'SparseCategoricalCrossentropy'},
                loss_weights={'classes': self.classes_weight},
              metrics=['accuracy'])
        self.history = self.model.fit(self.train_dataset, epochs=self.epochs, validation_data=self.val_dataset)
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
        self.trans_model.fit(self.train_dataset, epochs=2, validation_data=self.val_dataset)
        self.trans_model.save(r'./trained_models/pick_halo_transfer.h5')

    def show_CM(self):
        """显示混淆矩阵"""

        # load the model
        model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5')
        # rankdir='LR' is used to make the graph horizontal.
        tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
        # make predictions on the validation set

        X_val = self.X_test
        Y_val = self.Y_test
        y_pred = model.predict(X_val)
        y_pred = np.argmax(y_pred, axis=1)
        wrong_index = np.where(Y_val !=y_pred)
    
        wrong_data = X_val[wrong_index]

        cols  = [
                "ints_b_3",
                "ints_b_2",
                "ints_b_1",
                "ints_b0",
                "ints_b1",
                "ints_b2",
                "ints_b3",
                "m2_m1_10",
                "m1_m0_10",
                'b2_b1_10',
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
        # plot the confusion matrices, 1个子图
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        ConfusionMatrixDisplay.from_predictions(Y_val, y_pred, ax=axs, cmap=plt.cm.terrain)
        axs.set_title('Classifier')
    
        plt.show()
    
    def work_flow(self):
        """模型训练流程"""
        if not os.path.exists('./trained_models'):
            os.mkdir('./trained_models')
        self.load_dataset()
        self.get_model()
        # self.train()
        self.train_transfer()
        self.show_CM()

if __name__ == '__main__':
    pass