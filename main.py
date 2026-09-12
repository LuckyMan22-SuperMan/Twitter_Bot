import pandas as  pd

needed_columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "in_response_to_tweet_id",
    "response_tweet_id",
    "text"
]

df = pd.read_csv("twcs.csv", usecols=needed_columns)


#print(df.shape)
print(df.columns.tolist())
#print(df.head(4))

print(df.info())

print(df["author_id"].value_counts().head(20))

BRAND="AppleSupport"
brand_df=df[df["author_id"] == BRAND] #table of only "brand" responses

print(
    df[
        ["tweet_id", "author_id", "in_response_to_tweet_id", "response_tweet_id"]
    ].head(20)
)

tweet_lookup= df.set_index("tweet_id").to_dict("index")     #dict to find any tweet info

def get_conversation(tweet_id, tweet_lookup):
    conversations=[]
    current_id=tweet_id

    while pd.notna(current_id) and current_id in tweet_lookup:
        tweet=tweet_lookup[current_id]

        conversations.append(tweet)

        current_id=tweet["in_response_to_tweet_id"]
    conversations.reverse()

    return conversations
tweet_id=brand_df.iloc[0]["tweet_id"]   #get tweet id of 1st BRAND tweet
conversations=get_conversation(tweet_id,tweet_lookup)

for c in conversations:
    sender= "Customer" if c["inbound"] else BRAND
    #print(c["author_id"], ":", c["text"])
    print(sender, ":", c["text"])

# print("Total tweets:", len(df))
# print(f"{BRAND} tweets:", len(brand_df))

# print("LOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOL")
# customer_df = df[df["inbound"] == True]
# print(
#     customer_df[
#         ["tweet_id", "response_tweet_id", "text"]
#     ].head(20).to_string()
# )
def get_response_ids(value):
    if pd.isna(value):
        return []

    return [int(x.strip()) for x in str(value).split(",")] #print(get_response_ids("9,6,10")) gives a list with 9, 6, 10

customer_df = df[df["inbound"] == True]

for _, customer in customer_df.iterrows():

    if pd.notna(customer["response_tweet_id"]):
        print("CUSTOMER:")
        print(customer["text"])

        response_ids = get_response_ids(
            customer["response_tweet_id"]
        )

        print("RESPONSE IDS:", response_ids)

        for response_id in response_ids:
            response = tweet_lookup.get(response_id)

            if response is not None:
                print("AUTHOR:", response["author_id"])
                print("RESPONSE:", response["text"])

        break