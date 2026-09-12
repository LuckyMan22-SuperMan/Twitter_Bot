import pandas as pd

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

customer_tweet_id = interactions_df.iloc[0]["customer_tweet_id"]

conversation = get_conversation(customer_tweet_id)

print("\n" + "=" * 80)
print("FULL CONVERSATION")
print("=" * 80)

for tweet in conversation:
    sender = "Customer" if tweet["inbound"] else BRAND

    print(f"\n{sender}:")
    print(tweet["text"])