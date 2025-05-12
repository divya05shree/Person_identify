from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
dataframe = pd.read_csv("features.csv")

X_data = dataframe.iloc[:, :-1]
y_data = dataframe.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
joblib.dump(model, 'sleep_detection_model.joblib')

print("Model trained and saved successfully.")