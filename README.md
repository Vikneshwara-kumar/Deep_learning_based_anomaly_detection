# Deep Learning Based Anomaly Detection for Exoskeleton Assistance

## Project Description:
This project aims to develop a Cascade CNN-LSTM model for anomaly detection and mitigation in an exoskeleton system. The model combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to efficiently handle system anomalies, including sensor malfunctions, actuator failures, and communication errors. The goal is to detect faults before they occur, preventing and mitigating critical damage to the human wearing the exoskeleton.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Data Preprocessing](#data-preprocessing)
7. [Training the Model](#training-the-model)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

## Introduction
The Exoskeleton Assistance project aims to develop an untethered exoskeleton that provides real-world assistance. This exoskeleton is prone to various types of failures, such as sensor malfunctions, actuator failures, and communication errors. Detecting these faults before they occur can prevent and mitigate critical damage to the human wearing the exoskeleton. The anomaly detection system developed in this project leverages sensor data from the exoskeleton to identify potential issues.

## Project Structure
```
.
├── data
│   ├── raw
├── src
│   ├── data_preprocessing.py
│   ├── model_training.py
├── README.md
├── requirements.txt
└── setup_environment.sh

```

## Installation
### Prerequisites
*   Python 3.12
*   Git

### Steps
1.  Clone the repository:
```
git clone https://github.com/your-username/exoskeleton-anomaly-detection.git
cd exoskeleton-anomaly-detection
```

2. Create a virtual environment and install the dependencies:

```
./setup_environment.sh
```

## Usage
1.  Data Preprocessing:
    *   Upload the dataset in the data folder and the data in the excel file should be in this format First column **Time** and second column **Data**. 
    *   Run the data preprocessing script to prepare the data for training.

            python src/data_preprocessing.py
            
2.  Training the Model:
    *   Train the Cascade CNN-LSTM model using the preprocessed data.

            python src/model_training.py


## Model Architecture
The model consists of two main components:
*   Convolutional Neural Network (CNN): Extracts spatial features from the sensor data.
*   Long Short-Term Memory (LSTM): Captures temporal dependencies and sequences in the data.

##  Data Preprocessing
*   Load raw sensor data from the exoskeleton.
*   Handle missing values and normalize the data.
*   Split the data into training, validation, and test sets.

##  Training the Model
*   Define the CNN-LSTM model architecture.
*   Compile the model with appropriate loss functions and optimizers.
*   Train the model on the training dataset while validating on the validation set.

##  Evaluation
*   Assess the model's performance using metrics such as accuracy, precision, recall, and * F1-score.
*   Visualize the results and analyze the model's ability to detect anomalies.

##  Results
*   Summarize the key findings and performance metrics.
*   Include visualizations such as loss curves and confusion matrices.

##  Contributing
Contributions are welcome! Please follow these steps to contribute:

1.  Fork the repository.
2.  Create your feature branch (git checkout -b feature/AmazingFeature).
3.  Commit your changes (git commit -m 'Add some AmazingFeature').
4.  Push to the branch (git push origin feature/AmazingFeature).
5.   Open a pull request.

##  License
Distributed under the MIT License. See LICENSE for more information.

Acknowledgements
*   Inspired by advancements in deep learning and its applications in anomaly detection.