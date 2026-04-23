import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("churn.csv")

print("First 5 rows:")
print(data.head())


data = data.dropna()


data = pd.get_dummies(data, drop_first=True)


X = data.drop("Churn", axis=1)
y = data["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


while True:
    user_input = input("\nDo you want to test a customer? (yes/no): ")

    if user_input.lower() == "no":
        break

    try:
        print("\nEnter values:")

        tenure = int(input("Tenure (months): "))
        monthly_charges = float(input("Monthly Charges: "))
        total_charges = float(input("Total Charges: "))

        sample = [[tenure, monthly_charges, total_charges]]

        
        input_data = pd.DataFrame([[5, 90, 500]],
                columns=["tenure", "MonthlyCharges", "TotalCharges"])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            print(" Customer will CHURN")
        else:
            print(" Customer will STAY")

    except:
        print("Invalid input, try again!")