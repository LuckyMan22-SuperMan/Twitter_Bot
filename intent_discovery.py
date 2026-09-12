import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv("AppleSupport_conversations.csv")

print("Dataset shape:", df.shape)

texts = df["clean_customer_text"].fillna("")

valid_mask = texts.str.strip() != ""

texts = texts[valid_mask]

print("Usable customer messages:", len(texts))

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000
)

X = vectorizer.fit_transform(texts)

print("TF-IDF matrix shape:", X.shape)