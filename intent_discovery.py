import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv("AppleSupport_conversations.csv")

print("Dataset shape:", df.shape)