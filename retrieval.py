import pandas as pd
import re

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

df["clean_conversation"] = (
    df["clean_conversation"]
    .fillna("")
    .astype(str)
)

df["intent"] = df["intent"].fillna("other")


# Remove empty customer messages
df = df[
    df["clean_customer_text"].str.strip() != ""
].copy()

print("Usable historical messages:", len(df))


# ============================================
# 2. EXTRACT APPLESUPPORT RESPONSE
# ============================================

def extract_brand_response(conversation):

    """
    Extract the AppleSupport response from
    the historical conversation.
    """

    pattern = r"AppleSupport:\s*(.*?)(?=\s+Customer:|$)"

    matches = re.findall(
        pattern,
        conversation,
        flags=re.DOTALL
    )

    if not matches:
        return ""

    # Combine responses if AppleSupport replied multiple times
    response = " ".join(matches)

    return response.strip()

def is_useful_response(response):

    response = response.lower().strip()

    if not response:
        return False

    generic_phrases = [
        "support via twitter",
        "available in english",
        "preferred language",
        "contact us for help",
        "get help at",
        "join us",
        "follow us",
    ]

    for phrase in generic_phrases:
        if phrase in response:
            return False

    return True

df["brand_response"] = df["clean_conversation"].apply(
    extract_brand_response
)


# ============================================
# 3. KEEP ONLY CASES WITH A BRAND RESPONSE
# ============================================

retrieval_df = df[
    df["brand_response"].str.strip() != ""
].copy()

retrieval_df = retrieval_df[
    retrieval_df["brand_response"].apply(is_useful_response)
].copy()


print(
    "Cases with AppleSupport response:",
    len(retrieval_df)
)


# ============================================
# 4. CREATE TF-IDF
# ============================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(
    retrieval_df["clean_customer_text"]
)

print(
    "Retrieval TF-IDF shape:",
    X.shape
)


# ============================================
# 5. RETRIEVAL FUNCTION
# ============================================

def retrieve_similar_cases(
    query,
    predicted_intent,
    top_k=5
):

    # ----------------------------------------
    # Filter by intent
    # ----------------------------------------

    filtered_df = retrieval_df[
        retrieval_df["intent"] == predicted_intent
    ].copy()

    if len(filtered_df) == 0:
        return pd.DataFrame()


    # ----------------------------------------
    # Get TF-IDF vectors for filtered cases
    # ----------------------------------------

    filtered_indices = filtered_df.index

    filtered_positions = [
        retrieval_df.index.get_loc(index)
        for index in filtered_indices
    ]

    filtered_vectors = X[
        filtered_positions
    ]


    # ----------------------------------------
    # Convert query into TF-IDF
    # ----------------------------------------

    query_vector = vectorizer.transform(
        [query]
    )


    # ----------------------------------------
    # Calculate cosine similarity
    # ----------------------------------------

    similarities = cosine_similarity(
        query_vector,
        filtered_vectors
    )[0]


    # ----------------------------------------
    # Add similarity score
    # ----------------------------------------

    filtered_df["similarity"] = similarities


    # ----------------------------------------
    # Sort by similarity
    # ----------------------------------------

    results = filtered_df.sort_values(
        "similarity",
        ascending=False
    ).head(top_k)


    return results


# ============================================
# 6. TEST RETRIEVAL
# ============================================

query = "My iPhone battery is draining really fast"

predicted_intent = "battery_issue"

results = retrieve_similar_cases(
    query,
    predicted_intent,
    top_k=5
)


# ============================================
# 7. DISPLAY RESULTS
# ============================================

print("\nQuery:")
print(query)

print("\nPredicted intent:")
print(predicted_intent)

print("\nSimilar historical cases:")


if results.empty:

    print("No historical cases found.")

else:

    for _, row in results.iterrows():

        print("\n" + "=" * 50)

        print("Customer:")
        print(row["clean_customer_text"])

        print("\nSimilarity:")
        print(round(row["similarity"], 4))

        print("\nAppleSupport response:")
        print(row["brand_response"])