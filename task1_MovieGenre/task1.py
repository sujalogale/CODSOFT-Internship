import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



data = pd.read_csv("movies.csv")

print("First 5 rows:")
print(data.head())



data = data.dropna(subset=['plot', 'genre'])



X = data['plot']
y = data['genre']



tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)



model = MultinomialNB()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))



while True:
    user_input = input("\nEnter movie plot (or type 'exit'): ")

    if user_input.lower() == 'exit':
        break

    input_tfidf = tfidf.transform([user_input])
    prediction = model.predict(input_tfidf)

    print("Predicted Genre:", prediction[0])