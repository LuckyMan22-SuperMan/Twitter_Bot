import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv("AppleSupport_500_labeled_final.csv")

df["clean_customer_text"] = (
    df["clean_customer_text"]
    .fillna("")
    .astype(str)
)

df["intent"] = df["intent"].fillna("other")


# ============================================
# 2. REMOVE EMPTY MESSAGES
# ============================================

df = df[df["clean_customer_text"].str.strip() != ""].copy()

print("Usable historical messages:", len(df))


# ============================================
# 3. CREATE TF-IDF REPRESENTATION
# ============================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["clean_customer_text"])

print("TF-IDF matrix shape:", X.shape)


# ============================================
# 4. RETRIEVAL FUNCTION
# ============================================

def retrieve_similar_cases(
    query,
    predicted_intent,
    top_k=5
):

    # Filter by predicted intent
    filtered_df = df[
        df["intent"] == predicted_intent
    ].copy()

    if len(filtered_df) == 0:
        return pd.DataFrame()

    # Get vectors for the filtered cases
    filtered_indices = filtered_df.index

    filtered_vectors = X[
        [df.index.get_loc(i) for i in filtered_indices]
    ]

    # Convert query to TF-IDF
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_vector,
        filtered_vectors
    )[0]

    # Add similarity scores
    filtered_df["similarity"] = similarities

    # Sort by similarity
    results = filtered_df.sort_values(
        "similarity",
        ascending=False
    ).head(top_k)

    return results


# ============================================
# 5. TEST RETRIEVAL
# ============================================

query = "My iPhone battery is draining really fast"

predicted_intent = "battery_issue"

results = retrieve_similar_cases(
    query,
    predicted_intent,
    top_k=5
)


print("\nQuery:")
print(query)

print("\nPredicted intent:")
print(predicted_intent)

print("\nSimilar historical cases:")

for _, row in results.iterrows():

    print("\n--------------------------------")
    print("Customer:")
    print(row["clean_customer_text"])

    print("\nSimilarity:")
    print(round(row["similarity"], 4))

    print("\nHistorical conversation:")
    print(row["clean_conversation"])