
import string
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def clean_text(text):
    text = str(text).lower().translate(
        str.maketrans('', '', string.punctuation)
    )
    return " ".join(text.split())


# def predict_review(new_review, pipe):
#     cleaned = clean_text(new_review)

#     pred = pipe.predict([cleaned])[0]
#     prob = pipe.predict_proba([cleaned])[0]

#     confidence = prob[pred] * 100

#     label = "Deceptive" if pred == 1 else "Truthful"

#     if confidence < 60:
#         print(f"Review: {new_review[:100]}...")
#         print(f"Prediction: Uncertain ({label})")
#         print(f"Confidence: {confidence:.2f}%\n")
#     else:
#         print(f"Review: {new_review[:100]}...")
#         print(f"Prediction: {label}")
#         print(f"Confidence: {confidence:.2f}%\n")

def predict_review(new_review, pipe):
    cleaned = clean_text(new_review)

    pred = pipe.predict([cleaned])[0]
    prob = pipe.predict_proba([cleaned])[0]

    confidence = prob[pred] * 100

    label = "Deceptive" if pred == 1 else "Truthful"

    print(f"\nReview: {new_review}")

    if confidence < 60:
        print("Result: Uncertain")
        print(f"Most likely class: {label}")
    else:
        print(f"Result: {label}")

    print(f"Confidence: {confidence:.2f}%")


def main():

    url = "https://raw.githubusercontent.com/rajsiddarth/Text_Analytics/master/deceptive-opinion.csv"

    df = pd.read_csv(url)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    df = df.dropna(subset=['text', 'deceptive']).copy()

    df['label'] = df['deceptive'].map({
        'deceptive': 1,
        'truthful': 0
    })

    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)

    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000)),
        ("clf", LogisticRegression(
            random_state=42,
            max_iter=1000
        ))
    ])

    pipe.fit(X_train, y_train)

    vectorizer = pipe.named_steps["tfidf"]
    model = pipe.named_steps["clf"]

    y_pred = pipe.predict(X_test)

    print("\nModel Evaluation")

    print(
        f"Hold-out Accuracy: "
        f"{accuracy_score(y_test, y_pred) * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=['truthful', 'deceptive'],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )
    )

    print("\n5-Fold Cross-Validation")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring='accuracy'
    )

    print(
        "Cross-validation accuracies:",
        (scores * 100).round(2), "%"
    )

    print(
        f"Mean accuracy: {scores.mean() * 100:.2f}% "
        f"(+/- {scores.std() * 100:.2f} percentage points)"
    )

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    sorted_indices = np.argsort(model.coef_[0])

    print("\nTop 15 Truthful Words:")

    for i in sorted_indices[:15]:
        print(
            f"  {feature_names[i]:<15} : "
            f"{model.coef_[0][i]:.4f}"
        )

    print("\nTop 15 Deceptive Words:")

    for i in sorted_indices[-15:][::-1]:
        print(
            f"  {feature_names[i]:<15} : "
            f"{model.coef_[0][i]:.4f}"
        )

    print("\nTest Predictions")

    test_real = (
        "The room was a bit small and had a noisy AC unit, "
        "but the location on Michigan Ave was great."
    )

    test_fake = (
        "My family had the most luxurious vacation "
        "at this hotel!"
    )

    predict_review(test_real, pipe)
    predict_review(test_fake, pipe)

    print("\nHotel Review Prediction")

    while True:
        review = input(
            "\nEnter a hotel review (or type 'exit' to quit): "
        ).strip()

        if review.lower() == 'exit':
            break

        if not review:
            print("Please enter a valid review.")
            continue

        predict_review(review, pipe)

    print("\nProgram Execution Completed")


if __name__ == "__main__":
    main()




# import string
# import pandas as pd
# import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# from sklearn.model_selection import (
#     train_test_split,
#     StratifiedKFold,
#     cross_val_score
# )
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix
# )



# def clean_text(text):
#     text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
#     return " ".join(text.split())

# # def predict_review(new_review, vectorizer, model):
# #     cleaned = clean_text(new_review)
# #     vectorized = vectorizer.transform([cleaned])
# #     pred = model.predict(vectorized)[0]
# #     prob = model.predict_proba(vectorized)[0]
# #     label = "Deceptive" if pred == 1 else "Truthful"
# #     print(f"Review: {new_review[:100]}...")
# #     print(f"Prediction: {label} (Confidence: {prob[pred]*100:.2f}%)\n")


# def predict_review(new_review, pipe):
#     cleaned = clean_text(new_review)

#     pred = pipe.predict([cleaned])[0]
#     prob = pipe.predict_proba([cleaned])[0]

#     label = "Deceptive" if pred == 1 else "Truthful"

#     print(f"Review: {new_review[:100]}...")
#     print(
#         f"Prediction: {label} "
#         f"(Confidence: {prob[pred] * 100:.2f}%)\n"
#     )


# # load dataset
# url = "https://raw.githubusercontent.com/rajsiddarth/Text_Analytics/master/deceptive-opinion.csv"
# df = pd.read_csv(url)

# print("Dataset loaded successfully!")
# print("Dataset shape:", df.shape)

# # prepare target and features
# df['label'] = df['deceptive'].map({'deceptive': 1, 'truthful': 0})
# df['clean_text'] = df['text'].apply(clean_text)

# # vectorizer = TfidfVectorizer(max_features=3000)
# # X = vectorizer.fit_transform(df['clean_text'])
# # y = df['label']

# df = df.dropna(subset=['label', 'clean_text'])
# df['label'] = df['label'].astype(int)

# X = df['clean_text']
# y = df['label']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # train model
# # model = LogisticRegression(random_state=42)
# # model.fit(X_train, y_train)

# pipe = Pipeline([
#     ("tfidf", TfidfVectorizer(max_features=3000)),
#     ("clf", LogisticRegression(random_state=42, max_iter=1000))
# ])


# pipe.fit(X_train, y_train)

# vectorizer = pipe.named_steps["tfidf"]
# model = pipe.named_steps["clf"]

# # evaluate
# y_pred = pipe.predict(X_test)

# print("Model Evaluation")


# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print("\nClassification Report:")
# # print(classification_report(y_test, y_pred, target_names=['truthful', 'deceptive']))
# print(
#     classification_report(
#         y_test,
#         y_pred,
#         labels=[0, 1],
#         target_names=['truthful', 'deceptive'],
#         zero_division=0
#     )
# )

# print("Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred))
# print(
#     confusion_matrix(
#         y_test,
#         y_pred,
#         labels=[0, 1]
#     )
# )


# print("\n5-Fold Cross-Validation")

# cv = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )

# scores = cross_val_score(
#     pipe,
#     X,
#     y,
#     cv=cv,
#     scoring='accuracy'
# )

# print("Cross-validation accuracies:", scores.round(4))
# print(
#     f"Mean accuracy: {scores.mean():.4f} "
#     f"(+/- {scores.std():.4f})"
# )


# # top words associated with each class
# feature_names = np.array(vectorizer.get_feature_names_out())
# sorted_indices = np.argsort(model.coef_[0])

# print("\nTop 15 Truthful Words:")
# for i in sorted_indices[:15]:
#     print(f"  {feature_names[i]:<15} : {model.coef_[0][i]:.4f}")

# print("\nTop 15 Deceptive Words:")
# for i in sorted_indices[-15:][::-1]:
#     print(f"  {feature_names[i]:<15} : {model.coef_[0][i]:.4f}")

# # test custom reviews
# print("\nTest Predictions")
# test_real = "The room was a bit small and had a noisy AC unit, but the location on Michigan Ave was great."
# test_fake = "My family had the most luxurious vacation at this amazing hotel!"
# predict_review(test_real, pipe)
# predict_review(test_fake, pipe)



# print("\nHotel Review Prediction")

# while True:
#     review = input("\nEnter a hotel review (or type 'exit' to quit): ").strip()

#     if review.lower() == 'exit':
#         break

#     if not review:
#         print("Please enter a valid review.")
#         continue

#     predict_review(review, pipe)

# print("\nProgram Execution Completed")
