import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("hmn.csv")
X = df["Text"]
y = df["Label"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

joblib.dump(model, "toxic_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model trained and saved.")