import pandas as pd

df=pd.read_csv("AppleSupport_interactions.csv")

print(df.shape)
print(df.columns.to_list())

for i, row in df.head(20).iterrows():
    print("\n" + "=" * 80)
    print(f"INTERACTION {i + 1}")

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nAPPLE SUPPORT:")
    print(row["brand_response"])