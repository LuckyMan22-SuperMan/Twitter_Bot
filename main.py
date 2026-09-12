import pandas as  pd

df=pd.read_csv("twcs.csv")

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

tweet_lookup= df.set_index("tweet_id").to_dict("index")
