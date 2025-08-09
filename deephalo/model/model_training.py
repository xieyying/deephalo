import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from .training_data import prepare_training_dataset as create_dataset
from .model_build import isonn_model as model
from keras import layers
import keras_tuner as kt

class ElementPredicitonModel:
    '''
    A custom model class that includes methods for dataset loading, model building, training, and evaluation.

    Functions:
    load_dataset: Load the dataset.
    get_model: Build the model.
    train: Train the model.
    show_CM: Display the confusion matrix.
    work_flow: Complete model training workflow.
    '''

    def __init__(self, para) -> None:
        """
        Initialize model parameters. Parameters are passed as a dictionary for convenience.

        Args:
        para: class, model parameters, including batch_size, epochs, features, paths, test_path, classes, weight, learning_rate.
        """
        self.para = para
        self.input_shape = len(self.para.features_list)

    def load_dataset(self):
        """
        Load the dataset, create training, validation, and test sets.

        Returns:
        None
        """
        self.train_dataset, self.val_dataset, self.X_test, self.Y_test, self.val_ = create_dataset(
            self.para.features_list.copy(), self.para.paths, self.para.train_batch)

    def get_model(self):
        """
        Load the predefined model and plot the model structure.

        Returns:
        None
        """
        self.model = model(self.input_shape, self.para.classes)
        keras.utils.plot_model(self.model, to_file=r'./trained_models/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96)

    def train(self):
        """
        Train the model and save the trained model.

        Returns:
        None
        """
        opt = tf.keras.optimizers.Adam(learning_rate=self.para.learning_rate)

        self.model.compile(optimizer=opt,
                           loss={'classes': 'SparseCategoricalCrossentropy'},
                           loss_weights={'classes': self.para.classes_weight},
                           metrics=['accuracy'])

        # Create EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        self.history = self.model.fit(self.train_dataset, epochs=self.para.epochs,
                                      validation_data=self.val_dataset,
                                      callbacks=[early_stopping])  # Add callback to fit function
        self.model.summary()
        self.model.save(r'./trained_models/pick_halo_ann.h5')

        # Display the relationship between loss and epochs
        history = self.history.history
        print(history.keys())
        # Display different losses in different colors
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(history['loss'], color='red')
        plt.plot(history['accuracy'], color='blue')
        plt.plot(history['val_loss'], color='black')
        plt.plot(history['val_accuracy'], color='green')

        # Display legend
        plt.legend(['loss', 'accuracy', 'val_loss', 'val_accuracy'], loc='right')

    def show_cm(self):
        """
        Display the confusion matrix, calculate recall and precision, and save prediction results.

        Returns:
        None
        """
        # Load the model
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
        df_pred.to_csv(r'./trained_models/prediction.csv', index=False)

        y_pred = np.argmax(y_pred, axis=1)
        wrong_index = np.where(Y_val != y_pred)

        wrong_data = X_val[wrong_index]

        cols = [
            "p0_int",
            "p1_int",
            "p2_int",
            "p3_int",
            "p4_int",
            "mz_0",
            "m2_m1",
            "m1_m0",
        ]

        formula_test = np.array(self.val_['formula'].tolist())
        m0 = np.array(self.val_['mz_0'].tolist())
        wrong_data = pd.DataFrame(wrong_data, columns=cols)
        wrong_data['formula'] = pd.Series(formula_test[wrong_index])
        wrong_data['mz_0'] = pd.Series(m0[wrong_index])
        wrong_data['true_classes'] = pd.Series(Y_val[wrong_index])
        wrong_data['pred_classes'] = pd.Series(y_pred[wrong_index])

        wrong_data.to_csv(r'./trained_models/pick_halo_ann_wrong_data.csv', index=False)

        # Compute confusion matrix
        confusion_matrix(Y_val, y_pred)

        # Compute recall and precision for each class
        report = classification_report(Y_val, y_pred, output_dict=True, zero_division=0)
        recalls = [report[str(i)]['recall'] if str(i) in report else 0 for i in range(8)]
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
        # Annotate values on the chart
        for i, v in enumerate(precisions):
            axs[1, 0].text(i - 0.25, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')

        # Plot the recall bar chart to the right of the confusion matrix
        axs[0, 1].barh(np.arange(len(recalls)), recalls, color=colors)
        axs[0, 1].set_title('Recall')
        axs[0, 1].set_xlabel('Recall')
        axs[0, 1].set_ylabel('Class')
        # Annotate values on the chart
        for i, v in enumerate(recalls):
            axs[0, 1].text(v + 0.01, i + .25, str(round(v, 3)), color='black', fontweight='bold')
        axs[0, 1].invert_yaxis()  # Reverse the y-axis so class 0 is on top
        axs[0, 1].set_xlim(0, 1)  # Set the x-axis range to 0-1

        # Plot the f1-score bar chart below the recall bar chart
        axs[1, 1].bar(np.arange(len(F1_sore)), F1_sore, color=colors)
        axs[1, 1].set_title('F1-score')
        axs[1, 1].set_xlabel('Class')
        axs[1, 1].set_ylabel('F1-score')
        # Annotate values on the chart
        for i, v in enumerate(F1_sore):
            axs[1, 1].text(i - 0.25, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')
        axs[1, 1].set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def work_flow(self):
        """
        Model training workflow, including loading the dataset, building the model, training the model, and displaying the confusion matrix.

        Example:
        model = my_model(para)
        model.work_flow()
        """
        self.load_dataset()
        self.get_model()
        self.train()
        self.show_cm()

class HyperparameterSearch:
    """
    A custom class for hyperparameter search and model training using Keras Tuner.

    Functions:
    __init__: Initialize parameters.
    build_model: Build the model with hyperparameters.
    train_model: Train the model.
    perform_search: Perform hyperparameter search.
    """

    def __init__(self, input_shape, output_shape, loss_weights):
        """
        Initialize parameters.

        Args:
        input_shape: int, the dimension of input features.
        output_shape: int, the number of output classes.
        loss_weights: dict, loss weights for the model.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss_weights = loss_weights

    def build_model(self, hp):
        """
        Build the model with hyperparameters.

        Args:
        hp: kt.HyperParameters, the hyperparameter search space.

        Returns:
        model: keras.Model, the constructed model.
        """
        # Define the search space for batch size
        hp.Choice('batch_size', [4, 16, 64, 128], default=8)

        # Input layer
        inputs = layers.Input(shape=(self.input_shape,), name="features1")

        # Split input into two parts
        input1 = inputs[:, :-2]
        input2 = inputs[:, -2:]

        # Add Gaussian noise
        input1 = layers.GaussianNoise(hp.Choice('noise1', [0.03]))(input1)
        input2 = layers.GaussianNoise(hp.Choice('noise2', [0.001]))(input2)

        # Apply scaling
        power = hp.Choice('power_num', [0, 10, 20], default=0)
        input2 = tf.pow(input2, power)

        # Process features
        x = layers.Dense(hp.Choice("units_0x", [16, 64, 256], default=16), activation="relu")(input1)
        y = layers.Dense(hp.Choice("units_0y", [8, 32, 128, 512], default=8), activation="relu")(input2)
        share = layers.concatenate([x, y])

        # Define the number of layers in the neural network
        num_layers = hp.Int('num_layers', 1, 5)
        for i in range(num_layers):
            units = hp.Choice(f'units_{i+1}', [32, 128, 512], default=32)
            dropout = hp.Choice(f'dropout_{i+1}', [0.0, 0.3], default=0)
            share = layers.Dense(units, activation="relu")(share)
            share = layers.Dropout(dropout)(share)

        # Output layer
        clf_output = layers.Dense(self.output_shape, activation='softmax', name="classes")(share)

        # Compile the model
        model = Model(inputs=inputs, outputs=clf_output, name="pick_halo_ann")
        learning_rate = hp.Choice('learning_rate', [0.0001, 0.0003], default=0.0001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            loss_weights={'classes': self.loss_weights},
        )
        return model

    def train_model(self, hp, model, X_train, Y_train, X_test, Y_test, **kwargs):
        """
        Train the model.

        Args:
        hp: kt.HyperParameters, the hyperparameter search space.
        model: keras.Model, the constructed model.
        X_train: np.ndarray, training set features.
        Y_train: np.ndarray, training set labels.
        X_test: np.ndarray, validation set features.
        Y_test: np.ndarray, validation set labels.
        kwargs: dict, additional parameters.

        Returns:
        history: dict, training history.
        """
        batch_size = hp.get('batch_size')
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
        val_dataset = val_dataset.shuffle(len(X_test)).batch(batch_size)
        history = model.fit(train_dataset, validation_data=val_dataset, **kwargs)
        return history

    def perform_search(self, para, create_dataset_func):
        """
        Perform hyperparameter search.

        Args:
        para: class, model parameters.
        create_dataset_func: function, function to create datasets.

        Returns:
        None
        """
        # Load dataset
        paths = para.paths
        features = para.features_list
        input_shape = len(features)
        output_shape = para.classes
        classes_weight = para.classes_weight

        X_train, Y_train, X_test, Y_test = create_dataset_func(features.copy(), paths, para.train_batch, model='search')

        # Initialize the hypermodel
        hypermodel = HyperparameterSearch(input_shape, output_shape, classes_weight)

        # Set up the tuner
        tuner = kt.BayesianOptimization(
            hypermodel.build_model,
            objective="val_accuracy",
            max_trials=200,
            directory="0.001_inty_0.03_5_peaks",
            overwrite=False,
            project_name="kt_base",
            num_initial_points=5,
        )

        # Perform the search
        tuner.search(
            X_train,
            Y_train,
            X_test,
            Y_test,
            epochs=5,
            callbacks=[TensorBoard("./trained_models/tb_logs")],
        )
        tuner.results_summary()

if __name__ == '__main__':
    pass