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

# --------------------------------------------------
# Cluster customer messages
# --------------------------------------------------

NUM_CLUSTERS = 8

kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

print("\nCluster sizes:")
print(pd.Series(clusters).value_counts().sort_index())

# --------------------------------------------------
# Show important words for each cluster
# --------------------------------------------------

terms = vectorizer.get_feature_names_out()

print("\n" + "=" * 80)
print("IMPORTANT WORDS BY CLUSTER")
print("=" * 80)

for cluster_number in range(NUM_CLUSTERS):

    center = kmeans.cluster_centers_[cluster_number]

    top_indices = center.argsort()[-10:][::-1]

    top_words = terms[top_indices]

    print(
        f"\nCluster {cluster_number}:"
    )

    print(
        ", ".join(top_words)
    )