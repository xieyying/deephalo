import pandas as pd
import tensorflow as tf
import keras,os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .methods import create_dataset
from .model_build import model,model_sequence
class my_model:
    def __init__(self,para) -> None:
        self.batch_size = para['batch_size']
        self.epochs = para['epochs']
        self.features = para['features']
        self.paths = para['paths']
        self.input_shape = len(self.features)
        self.output_shape = para['classes']
        self.classes_weight = para['weight']
        self.learning_rate = para['learning_rate']
    def load_dataset(self):
        self.train_dataset,self.val_dataset,self.X_test, self.Y_test,self.val_ = create_dataset(self.features,self.paths,self.batch_size)

    def get_model(self):
        #model_build中可以定义多种模型结构方便切换
        self.model = model(self.input_shape,self.output_shape)
        #绘制模型图
        keras.utils.plot_model(self.model, to_file=r'./trained_models/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96)
    
    def train(self):
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

    def show_CM(self):
        # load the model
        model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5')

        # make predictions on the validation set

        X_val = self.X_test
        Y_val = self.Y_test
        y_pred = model.predict(X_val)
        y_pred = np.argmax(y_pred, axis=1)
        wrong_index = np.where(Y_val !=y_pred)
    
        wrong_data = X_val[wrong_index]

        cols  = [
                "ints_b3",
                "ints_b2",
                "ints_b1",
                "ints_a0",
                "ints_a1",
                "ints_a2",
                "ints_a3",
                "new_a2_a1_10",
            ]
           
        formula_test = np.array(self.val_['formula'].tolist())
        new_a0 = np.array(self.val_['new_a0_mz'].tolist())
        wrong_data = pd.DataFrame(wrong_data,columns=cols)
        wrong_data['formula'] = pd.Series(formula_test[wrong_index])
        wrong_data['new_a0_mz'] = pd.Series(new_a0[wrong_index])
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
        if not os.path.exists('./trained_models'):
            os.mkdir('./trained_models')
        self.load_dataset()
        self.get_model()
        self.train()
        self.show_CM()

if __name__ == '__main__':
    para = {
            'batch_size': 1000,
            'epochs': 10,
            'features': ["new_a0_ints","new_a1_ints","new_a2_ints","new_a3_ints","new_a2_a1","new_a2_a0"],
            'paths':    [r'C:\Users\Xin\Desktop\p_test\train_dataset\selected_add_Fe_data.csv',
                         r'C:\Users\Xin\Desktop\p_test\train_dataset\selected_data.csv',
                         r'C:\Users\Xin\Desktop\p_test\train_dataset\selected_hydroisomer_data.csv',
                         r'C:\Users\Xin\Desktop\p_test\train_dataset\selected_hydroisomer2_data.csv',
                         r'C:\Users\Xin\Desktop\p_test\train_dataset\selected_hydroisomer3_data.csv',],
            'base_classes':3,
            'sub_classes':5,
            'hydro_classes':8,
            }
    model = my_model(para)
    model.work_flow()
