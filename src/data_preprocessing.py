import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, file_path, time_steps=50, subsequences=2):
        self.file_path = file_path
        self.time_steps = time_steps
        self.subsequences = subsequences
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        # Load the dataset
        self.df = pd.read_excel(self.file_path, parse_dates=['Time'])

        # Plot the data
        self.df.plot(x='Time', y='Data', label='Data')
        plt.show()

    def split_data(self):
        # Split the data into training and testing sets
        train_size = int(len(self.df) * 0.8)
        test_size = len(self.df) - train_size
        self.train_data, self.test_data = self.df.iloc[0:train_size], self.df.iloc[train_size:len(self.df)]
        print(f"Train set shape: {self.train_data.shape}")
        print(f"Test set shape: {self.test_data.shape}")

    def standardize_data(self):
        # Standardize the dataset
        scaler = StandardScaler()
        scaler = scaler.fit(self.train_data[['Data']])
        self.train_data['Data'] = scaler.transform(self.train_data[['Data']])
        self.test_data['Data'] = scaler.transform(self.test_data[['Data']])

    def create_sequences(self, data):
        Xs, ys = [], []
        for i in range(len(data) - self.time_steps):
            Xs.append(data.iloc[i:(i + self.time_steps)].values)
            ys.append(data.iloc[i + self.time_steps])
        return np.array(Xs), np.array(ys)

    def reshape_data(self):
        # Create sequences
        self.X_train, self.y_train = self.create_sequences(self.train_data[['Data']])
        self.X_test, self.y_test = self.create_sequences(self.test_data[['Data']])

        # Reshape the input data to be 3-dimensional
        timesteps = self.X_train.shape[1] // self.subsequences
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.subsequences, timesteps, 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.subsequences, timesteps, 1))

        print(f'Train set shape: {self.X_train.shape}')
        print(f'Test set shape: {self.X_test.shape}')

    def save_data(self):
        # Save preprocessed data to disk
        np.save('X_train.npy', self.X_train)
        np.save('y_train.npy', self.y_train)
        np.save('X_test.npy', self.X_test)
        np.save('y_test.npy', self.y_test)

    def process(self):
        self.load_data()
        self.split_data()
        self.standardize_data()
        self.reshape_data()
        self.save_data()

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor('Sensor_data.xlsx')
    preprocessor.process()
