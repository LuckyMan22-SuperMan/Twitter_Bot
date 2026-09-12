import pandas as pd
import re

def clean_text(text):
    text = re.sub(r'@\w+', '', text)       # remove @mentions
    text = re.sub(r'https?://\S+', '', text)  # remove URLs
    text = re.sub(r'\s+', ' ', text)       # normalize spaces
    return text.strip()

BRAND = "AppleSupport"

needed_columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "in_response_to_tweet_id",
    "response_tweet_id",
    "text"
]

# Load raw dataset
df = pd.read_csv("twcs.csv", usecols=needed_columns)

# Load our 500 interactions
interactions_df = pd.read_csv("AppleSupport_interactions.csv")

# Create lookup dictionary
tweet_lookup = df.set_index("tweet_id").to_dict("index")


def get_conversation(tweet_id):
    conversation = []
    current_id = tweet_id

    while pd.notna(current_id) and current_id in tweet_lookup:
        tweet = tweet_lookup[current_id]

        conversation.append(tweet)

        current_id = tweet["in_response_to_tweet_id"]

    conversation.reverse()

    return conversation

conversations = []

for _, row in interactions_df.iterrows():

    customer_tweet_id = row["customer_tweet_id"]

    conversation = get_conversation(customer_tweet_id)

    conversations.append({
        "customer_tweet_id": customer_tweet_id,
        "brand_tweet_id": row["brand_tweet_id"],
        "conversation": conversation
    })

print("Number of conversations:", len(conversations))
conversation_rows = []

for item in conversations:

    messages = []

    for tweet in item["conversation"]:

        sender = "Customer" if tweet["inbound"] else BRAND

        messages.append(
            f"{sender}: {tweet['text']}"
        )

    conversation_text = "\n".join(messages)

    conversation_rows.append({
        "customer_tweet_id": item["customer_tweet_id"],
        "brand_tweet_id": item["brand_tweet_id"],
        "conversation_text": conversation_text
    })

conversations_df = pd.DataFrame(conversation_rows)

conversations_df["clean_conversation"] = (
    conversations_df["conversation_text"]
    .apply(clean_text)
)

print("\nConversation dataset shape:", conversations_df.shape)
print("\nColumns:", conversations_df.columns.tolist())

print("\n" + "=" * 80)
print("FIRST CONVERSATION")
print("=" * 80)
print(conversations_df.iloc[0]["conversation_text"])

conversations_df.to_csv(
    "AppleSupport_conversations.csv",
    index=False
)

print("\nSaved:", len(conversations_df), "conversations")

print("\n" + "=" * 80)
print("BASIC DATA CHECKS")
print("=" * 80)

print("\nTotal conversations:", len(conversations_df))

print("\nMissing values:")
print(conversations_df.isna().sum())

print("\nDuplicate conversations:")
print(conversations_df["conversation_text"].duplicated().sum())

print("\nConversation lengths:")
print(
    conversations_df["conversation_text"]
    .str.len()
    .describe()
)

shortest = conversations_df.sort_values(
    by="clean_conversation",
    key=lambda x: x.str.len()
).head(15)

print("\n" + "=" * 80)
print("10 SHORTEST CONVERSATIONS")
print("=" * 80)

for _, row in shortest.iterrows():
    print("\n" + "-" * 80)
    print(row["clean_conversation"])