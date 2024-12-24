import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler

# Load the dataset
data={
    'Temperature':[30,22,25,28,35,18,21,29,32,24],
    'Humidity':[70,65,80,75,60,85,90,68,72,88],
    'WindSpeed':[10,5,15,12,8,6,7,14,10,9],
    'Condition':['Sunny','Cloudy','Rainy','Sunny','Sunny','Rainy','Rainy','Cloudy','Sunny','Rainy']
}

df=pd.DataFrame(data)

#preprocess the data
label_encoder=LabelEncoder()
df['Condition'] =label_encoder.fit_transform(df['Condition'])

# Separate the features and target
X=df[['Temperature','Humidity','WindSpeed']]
y=df['Condition']

# scale numerical features

scaler=StandardScaler()
X=scaler.fit_transform(X)



# Split into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# train the model
model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

# evaluate the model

y_pred=model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Classfication Report: \n",classification_report(y_test,y_pred))


# Make predictions with a new sample
sample_data = pd.DataFrame([[33, 65, 12]], columns=['Temperature', 'Humidity', 'WindSpeed'])  # Use DataFrame with column names
sample_data_scaled = scaler.transform(sample_data)  # Transform the new sample data

prediction = model.predict(sample_data_scaled)

# Decode the prediction back to the original weather condition label
print("Predicted Weather Condition:", label_encoder.inverse_transform(prediction))