import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("creditcard.csv")

print("First 5 rows:")
print(data.head())


print("\nMissing values:\n", data.isnull().sum())


X = data.drop("Class", axis=1)   
y = data["Class"]                


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)



print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


while True:
    user_input = input("\nDo you want to test a transaction? (yes/no): ")

    if user_input.lower() == "no":
        break

    sample = X_test.iloc[[0]]   
    prediction = model.predict(sample)

    if prediction[0] == 1:
        print(" Fraud Transaction Detected")
    else:
        print(" Legitimate Transaction")