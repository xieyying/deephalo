import tensorflow as tf
import numpy as np
import datetime
from .model_base import base
from sklearn.metrics import  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle,os
import keras
from keras.utils import plot_model
from keras import layers
import pandas as pd
np.set_printoptions(suppress=True)
class pick_halo_ann(base):
    """
    ann_para:dense1,dense1_drop,dense2,dense2_drop,classes
    """
    def create_Dataset(self):     
        if self.use_noise_data == 'True' or self.use_add_fe_data == 'True' or self.use_hydroisomer_data == 'True':
            train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, (self.Y_train,self.X_train_sub_group,self.X_train_hydro_group)))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, (self.Y_train,self.X_train_sub_group,self.X_train_hydro_group)))
        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, (self.Y_test,self.X_test_sub_group,self.X_test_hydro_group)))
        train_dataset = train_dataset.shuffle(len(self.X_train)).batch(self.train_batch)
        val_dataset = val_dataset.shuffle(len(self.X_test)).batch(self.val_batch)
        return train_dataset,val_dataset


    def train(self):
        #创建训练集和验证集
        train_dataset,val_dataset = self.create_Dataset()
        #模型参数设置
        def get_compiled_model():
            input = keras.Input(shape=(6,), name="mass_features")
            share = layers.Dense(4000, activation="relu")(input)
            share = layers.Dropout(0.5)(share)
            share = layers.Dense(1000, activation="relu")(share)
            share = layers.Dropout(0.3,name='share')(share)

            x = layers.Dense(500, activation="relu")(share)
            clf_base_output = layers.Dense(3,activation='softmax', name="base")(x)

            y = layers.Concatenate()([share, clf_base_output])
            y = layers.Dense(500, activation="relu")(y)
            clf_sub_output = layers.Dense(4,activation='softmax', name="sub")(y)

            z2 = layers.Concatenate()([share, clf_base_output, clf_sub_output])
            z2 = layers.Dense(500, activation="relu")(z2)
            clf_hydro_output = layers.Dense(7,activation='softmax', name="hydroisomer")(z2)

            clfs = keras.Model(inputs=input, outputs=[clf_base_output, clf_sub_output,  clf_hydro_output], name="clfs")
            
            clfs.compile(   optimizer='adam',
                            loss={'base': 'SparseCategoricalCrossentropy',
                                  'sub': 'SparseCategoricalCrossentropy',
                                  'hydroisomer': 'SparseCategoricalCrossentropy'},
                            loss_weights={'base': self.parameters['base_weight'],
                                            'sub': self.parameters['sub_weight'],
                                            'hydroisomer': self.parameters['hydroisomer_weight']},
                            metrics=['accuracy']
                        )
            return clfs
        #待训练模型
        self.clf = get_compiled_model()
        #记录训练过程
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #训练评估
        # self.clf.fit(train_dataset, epochs=self.parameters['epochs'],class_weight=self.parameters['class_weight'],callbacks=[tensorboard_callback])
        self.clf.fit(train_dataset, epochs=self.parameters['epochs'],callbacks=[tensorboard_callback])
        self.clf.evaluate(val_dataset)

        #训练结束的模型保存至本地文件中
        if self.save == True:
            if not os.path.exists(r'./trained_models'):
                os.makedirs(r'./trained_models')
            self.clf.save(r'./trained_models/pick_halo_ann.h5')


    
    def show_CM(self,pre_trained = False):#查看混淆矩阵
        if pre_trained == True:
            self.clf = tf.keras.models.load_model(r'./trained_models/pick_halo_ann.h5')
        #若模型未训练
        elif self.clf == None:
            self.train()
        #获得真实标签
        # t = self.Y_test
        y_val = self.Y_test
        sub_val = self.X_test_sub_group
        hydro_val = self.X_test_hydro_group

        #获得预测结果
        pr = self.clf.predict(self.X_test)
        pr_base = np.argmax(pr[0],axis=1)
        pr_sub = np.argmax(pr[1],axis=1)
        pr_hydro = np.argmax(pr[2],axis=1)


 


        #绘制混淆矩阵图
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        base_cm = ConfusionMatrixDisplay.from_predictions(y_val, pr_base,ax=axs[0],  cmap=plt.cm.terrain)
        axs[0].set_title('Classifier 1')
        sub_cm = ConfusionMatrixDisplay.from_predictions(sub_val, pr_sub, ax=axs[1], cmap=plt.cm.terrain)
        axs[1].set_title('Classifier 2')
        hydro_cm = ConfusionMatrixDisplay.from_predictions(hydro_val, pr_hydro,ax=axs[2], cmap=plt.cm.terrain)
        axs[2].set_title('Classifier 3')
        plt.show()

        if not os.path.exists(r'./trained_models'):
            os.makedirs(r'./trained_models')
        self.model_sum = plot_model(self.clf,to_file=r'./trained_models/pick_halo_ann.png',show_shapes=True)#,rankdir='LR')
        pickle.dump([fig,self.parameters],open(r'./trained_models/pick_halo_ann_sum.pkl','wb'))
        #将真实标签和预测结果不一致的数据保存至本地文件中
        wrong_index = np.where((y_val != pr_base)|(sub_val != pr_sub)|(hydro_val != pr_hydro))
        
        wrong_data = self.X_test[wrong_index]
        #delete 'group','formula' column from self.features
        cols = self.features

        wrong_data = pd.DataFrame(wrong_data,columns=cols)
        wrong_data['formula'] = pd.Series(self.formula_test[wrong_index])   
        wrong_data['true_base'] = pd.Series(y_val[wrong_index])
        wrong_data['pred_base'] = pd.Series(pr_base[wrong_index])
        wrong_data['true_sub'] = pd.Series(sub_val[wrong_index])
        wrong_data['pred_sub'] = pd.Series(pr_sub[wrong_index])
        wrong_data['true_hydro'] = pd.Series(hydro_val[wrong_index])
        wrong_data['pred_hydro'] = pd.Series(pr_hydro[wrong_index])



        #将wrong_data转为dataframe

        wrong_data.to_csv(r'./trained_models/pick_halo_ann_wrong_data.csv',index=False)
        