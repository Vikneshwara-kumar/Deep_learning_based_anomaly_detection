import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam

class CNNLSTMModel:
    def __init__(self, learning_rate=0.001, epochs=40, batch_size=256):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.model = None

    def build_model(self, input_shape):
        model = keras.Sequential()
        model.add(TimeDistributed(layers.Conv1D(64, 3, activation='relu', input_shape=input_shape)))
        model.add(TimeDistributed(layers.MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(layers.Flatten()))
        model.add(layers.LSTM(16, activation='tanh'))
        model.add(layers.Dense(16))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer=self.optimizer)
        self.model = model

    def train_model(self, X_train, y_train):
        history = self.model.fit(
            X_train, y_train, validation_split=0.1, shuffle=False,
            epochs=self.epochs, batch_size=self.batch_size, verbose=1,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')]
        )
        return history

    def evaluate_model(self, X, y):
        predictions = self.model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        return rmse, r2, predictions

    def save_model(self, model_path):
        self.model.save(model_path)

    def save_logs(self, log_path, train_rmse, train_r2, test_rmse, test_r2, Inference_time):
        with open(log_path, 'w') as log_file:
            log_file.write(f'Train RMSE: {train_rmse*100}\n')
            log_file.write(f'Train R2 Score: {train_r2*100}\n')
            log_file.write(f'Test RMSE: {test_rmse*100}\n')
            log_file.write(f'Test R2 Score: {test_r2*100}\n')
            log_file.write(f'Inference Time: {Inference_time}\n')
            

    def measure_inference_time(self, X_sample):
        start_time = time.time()
        _ = self.model.predict(X_sample)
        end_time = time.time()
        inference_time = end_time - start_time
        return inference_time

def main():
    # Load preprocessed data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Initialize and build the model
    model = CNNLSTMModel(learning_rate=0.001, epochs=40, batch_size=256)
    model.build_model(input_shape=(X_train.shape[2], X_train.shape[3]))

    # Train the model
    history = model.train_model(X_train, y_train)

    # Plot training and validation loss over epochs
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')
    plt.show()

    # Evaluate the model on training and test sets
    train_rmse, train_r2, _ = model.evaluate_model(X_train, y_train)
    test_rmse, test_r2, test_predictions = model.evaluate_model(X_test, y_test)

    print(f'Train RMSE: {train_rmse}')
    print(f'Train R2 Score: {train_r2}')
    print(f'Test RMSE: {test_rmse}')
    print(f'Test R2 Score: {test_r2}')

    # Measure inference time
    inference_time = model.measure_inference_time(X_test[:32])
    print(f'Inference time for a batch of 32 samples: {inference_time} seconds')

    # Save the model
    model.save_model("model_cnn_lstm.h5")

    # Save logs
    model.save_logs('model_logs.txt', train_rmse, train_r2, test_rmse, test_r2,inference_time)

if __name__ == "__main__":
    main()
