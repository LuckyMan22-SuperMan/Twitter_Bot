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

# --------------------------------------------------
# Show example customer messages from each cluster
# --------------------------------------------------

df_valid = df[valid_mask].copy()

df_valid["cluster"] = clusters

print("\n" + "=" * 80)
print("EXAMPLE MESSAGES BY CLUSTER")
print("=" * 80)

for cluster_number in range(NUM_CLUSTERS):

    print(
        f"\n\n{'=' * 30} CLUSTER {cluster_number} {'=' * 30}"
    )

    examples = df_valid[
        df_valid["cluster"] == cluster_number
    ]["clean_customer_text"].head(8)

    for example in examples:
        print("-", example)

        # --------------------------------------------------
# Save clustered messages for manual inspection
# --------------------------------------------------

df_valid = df[valid_mask].copy()

df_valid["cluster"] = clusters

inspection_df = df_valid[
    [
        "customer_tweet_id",
        "clean_customer_text",
        "cluster"
    ]
].reset_index(drop=True)

inspection_df.insert(
    0,
    "index",
    inspection_df.index
)

inspection_df.to_csv(
    "AppleSupport_intent_inspection.csv",
    index=False
)

print(
    "\nSaved intent inspection file:",
    len(inspection_df),
    "messages"
)

print("\n" + "=" * 80)
print("MESSAGES PER CLUSTER")
print("=" * 80)

print(
    inspection_df["cluster"]
    .value_counts()
    .sort_index()
)
for cluster_number in range(8):

    print("\n" + "=" * 80)
    print(f"CLUSTER {cluster_number}")
    print("=" * 80)

    cluster_messages = inspection_df[
        inspection_df["cluster"] == cluster_number
    ]["clean_customer_text"]

    for message in cluster_messages:
        print("-", message)