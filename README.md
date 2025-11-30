Human Activity Recognition Using Machine Learning

This project identifies physical human activities using smartphone accelerometer and gyroscope sensor data. Multiple machine learning models were trained and compared to classify activities such as walking, sitting, standing, and other motion patterns using the UCI HAR Dataset.

Project Overview

The goal of this project is to build an intelligent classification system capable of recognizing human physical activities based on motion sensor data from mobile devices.

This project includes:

Data preprocessing and feature scaling

Exploratory data analysis

Training multiple machine learning and deep learning models

Model evaluation and comparison

Saving trained models for future use and deployment

Dataset Information

Dataset: UCI Human Activity Recognition Using Smartphones

Source: UCI Machine Learning Repository

Sensors: Accelerometer and Gyroscope

Activity Labels
Label	Activity
1	Walking
2	Walking Upstairs
3	Walking Downstairs
4	Sitting
5	Standing
6	Laying
Technologies and Tools
Category	Tools Used
Programming Language	Python
Libraries	NumPy, Pandas, Scikit-Learn, TensorFlow, Matplotlib, Seaborn
Environment	Google Colab
Model Storage	Joblib and TensorFlow SavedModel format
Models Trained and Evaluated

Three models were trained and compared:

Random Forest Classifier

Support Vector Machine (SVM)

Deep Neural Network (TensorFlow)

Techniques Applied

StandardScaler normalization

Hyperparameter tuning using RandomizedSearchCV

Evaluation using accuracy, classification report, and confusion matrix

Results
Model	Accuracy
Random Forest	92.77%
SVM	95.52%
Neural Network	92.94%

Best Model Accuracy achieved: 95.52% using Support Vector Machine (SVM)

This result was validated using confusion matrix analysis, classification reports, and cross-model comparison.

Saved Model Files
File Name	Description
HAR_RF_Model.pkl	Trained Random Forest model
HAR_SVM_Model.pkl	Trained Support Vector Machine model
HAR_NN_Model.h5	Trained Neural Network model
HAR_Scaler.pkl	StandardScaler instance for preprocessing
Sample Prediction Code
sample = X_test_scaled[10].reshape(1, -1)
prediction = rf_best.predict(sample)
print("Predicted Activity:", activity_labels.loc[prediction+1].values[0])

Project Structure
HAR-Project
 ┣ data
 ┣ models
 ┣ HAR_Training.ipynb
 ┣ README.md
 ┣ requirements.txt

Future Improvements

Implement LSTM or CNN models to better model sequential sensor data

Deploy the system using Streamlit, Flask, or FastAPI

Develop a mobile-friendly version using TensorFlow Lite

Enable real-time predictions using live sensor streams

Acknowledgements

Dataset sourced from the UCI Machine Learning Repository:
Human Activity Recognition Using Smartphones Dataset.
