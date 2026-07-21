import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def clean_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def predict_review(new_review, vectorizer, model):
    cleaned = clean_text(new_review)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    label = "Deceptive" if pred == 1 else "Truthful"
    print(f"Review: {new_review[:60]}...")
    print(f"Prediction: {label} (Confidence: {prob[pred]*100:.2f}%)\n")

# load dataset
url = "https://raw.githubusercontent.com/rajsiddarth/Text_Analytics/master/deceptive-opinion.csv"
df = pd.read_csv(url)

# prepare target and features
df['label'] = df['deceptive'].map({'deceptive': 1, 'truthful': 0})
df['clean_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['truthful', 'deceptive']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# top words associated with each class
feature_names = np.array(vectorizer.get_feature_names_out())
sorted_indices = np.argsort(model.coef_[0])

print("\nTop 15 Truthful Words:")
for i in sorted_indices[:15]:
    print(f"  {feature_names[i]:<15} : {model.coef_[0][i]:.4f}")

print("\nTop 15 Deceptive Words:")
for i in sorted_indices[-15:][::-1]:
    print(f"  {feature_names[i]:<15} : {model.coef_[0][i]:.4f}")

# test custom reviews
print("\nTest Predictions")
test_real = "The room was a bit small and had a noisy AC unit, but the location on Michigan Ave was great."
test_fake = "My family had the most luxurious vacation at this amazing hotel!"
predict_review(test_real, vectorizer, model)
predict_review(test_fake, vectorizer, model)
