Weather Prediction with Random Forest Classifier

Description

This project uses machine learning to predict weather conditions (such as "Sunny", "Rainy", or "Cloudy") based on features like temperature, humidity, and wind speed. The model is built using a Random Forest Classifier and data preprocessing steps, including Label Encoding for categorical variables and Standard Scaling for numerical features.

Features

Data Preprocessing:

Label encoding is used to convert weather conditions (text labels) into numeric labels.

StandardScaler is used to scale the numerical features (Temperature, Humidity, and WindSpeed) to standardize their values.

Model Training: A Random Forest Classifier is used to train the model on the dataset and make predictions on unseen data.

Evaluation: The model is evaluated using accuracy and a classification report to assess its performance.

Prediction: After training, the model can predict the weather condition for new input data.
