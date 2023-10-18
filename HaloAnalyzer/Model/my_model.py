import tensorflow as tf
import keras
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
        self.output1_shape = para['base_classes']
        self.output2_shape = para['sub_classes']
        self.output3_shape = para['hydro_classes']
        self.base_weight = para['base_weight']
        self.sub_weight = para['sub_weight']
        self.hydroisomer_weight = para['hydroisomer_weight']
        self.learning_rate = para['learning_rate']
    def load_dataset(self):
        self.train_dataset,self.val_dataset = create_dataset(self.features,self.paths,self.batch_size)

    def get_model(self):
        #model_build中可以定义多种模型结构方便切换
        self.model = model_sequence(self.input_shape,self.output1_shape,self.output2_shape,self.output3_shape)
        #绘制模型图
        keras.utils.plot_model(self.model, to_file=r'./trained_models/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96)
    
    def train(self):
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=opt,
              loss={'base': 'SparseCategoricalCrossentropy',
                    'sub': 'SparseCategoricalCrossentropy',
                    'hydro': 'SparseCategoricalCrossentropy'},
                loss_weights={'base': self.base_weight,
                                'sub': self.sub_weight,
                                'hydro': self.hydroisomer_weight},
              metrics=['accuracy'])
        self.history = self.model.fit(self.train_dataset, epochs=self.epochs, validation_data=self.val_dataset)
        self.model.save(r'./trained_models/pick_halo_ann.h5')

    def show_CM(self):
        # load the model
        model = keras.models.load_model(r'./trained_models/pick_halo_ann.h5')

        # make predictions on the validation set
        val_data = next(iter(self.val_dataset))
        X_val, (Y_val, sub_group,hydro_group) = val_data
        y_pred = model.predict(X_val)

        # plot the confusion matrices
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        ConfusionMatrixDisplay.from_predictions(Y_val, np.argmax(y_pred[0], axis=1), ax=axs[0], cmap=plt.cm.terrain)
        axs[0].set_title('Base Classifier')
        ConfusionMatrixDisplay.from_predictions(sub_group, np.argmax(y_pred[1], axis=1), ax=axs[1], cmap=plt.cm.terrain)
        axs[1].set_title('Sub Classifier')
        ConfusionMatrixDisplay.from_predictions(hydro_group, np.argmax(y_pred[2], axis=1), ax=axs[2], cmap=plt.cm.terrain)
        axs[2].set_title('Hydro Classifier')
        plt.show()
    
    def work_flow(self):
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
