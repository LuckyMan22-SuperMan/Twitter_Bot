import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv("AppleSupport_50_labeled_final.csv")

print("Dataset shape:", df.shape)
print("\nIntent distribution:")
print(df["intent"].value_counts())


# ============================================
# 2. REMOVE EMPTY MESSAGES
# ============================================

df["clean_customer_text"] = df["clean_customer_text"].fillna("")

df = df[df["clean_customer_text"].str.strip() != ""]

print("\nUsable messages:", len(df))


# ============================================
# 3. SPLIT DATA
# ============================================

X = df["clean_customer_text"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 4. TF-IDF + LOGISTIC REGRESSION
# ============================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# ============================================
# 5. TRAIN
# ============================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================
# 6. PREDICT
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 7. EVALUATE
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ============================================
# 8. CONFUSION MATRIX
# ============================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=labels,
    columns=labels
))


# ============================================
# 9. TEST WITH NEW CUSTOMER MESSAGES
# ============================================

new_messages = [
    "My iPhone battery is draining really fast",
    "My keyboard keeps changing the letters I type",
    "My WiFi is not connecting",
    "My apps keep crashing",
    "My iPhone became very slow after the update"
]

predictions = model.predict(new_messages)
probabilities = model.predict_proba(new_messages)

print("\n\nNew Message Predictions:")

for message, prediction, probability in zip(
    new_messages,
    predictions,
    probabilities
):
    confidence = probability.max()

    print("\nMessage:", message)
    print("Predicted intent:", prediction)
    print("Confidence:", round(confidence, 4))