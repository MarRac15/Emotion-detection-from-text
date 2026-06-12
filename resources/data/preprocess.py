import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import random
import re
from sklearn.preprocessing import LabelEncoder


RAW_DATA_PATH = "resources/data/raw/tweet_emotions.csv"


data = pd.read_csv(RAW_DATA_PATH)

#Examine the datatypes
# print(data.info())
# print(data.head())

#Check what are the classes in this dataset:
# print(pd.unique(data["sentiment"]))

#Check for nulls, Nans and duplicates:
print(f"liczba wszystkich wartości NaN w całej tabeli: {data.isna().sum().sum()}")
print(f"liczba wszystkich wartości null w całej tabeli: {data.isnull().sum()}")
print(f"Czy są jakieś duplikaty wpisów: {data.duplicated().sum()}")

#Lets see the distribution of the classes:
print(data["sentiment"].value_counts())
plt.figure(figsize=(10, 5))
data['sentiment'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


#Lets cut some of them (empty, anger, boredom, enthusiasm)
classes_to_keep = [
    "neutral",
    "worry",
    "happiness",
    "sadness",
    "love",
    "surprise",
    "fun",
    "relief",
    "hate"
]

new_data = data[data["sentiment"].isin(classes_to_keep)].copy()


#labels to id:
encoder = LabelEncoder()
new_data["label_id"] = encoder.fit_transform(new_data["sentiment"])
print("\nRozkład klas po zbalansowaniu: ")
print(new_data["sentiment"].value_counts().sort_values(ascending=False))
print(round(new_data["sentiment"].value_counts(normalize=True) * 100, 2))


#CZY usuwac apostrofy?
def clean_text(dirty_text: str):
    punctuation = '!?.,;:()[]{}"'
    new_str = dirty_text.lower()
    
    #url
    new_str = re.sub(r"http\S+|www\S+", "", new_str)

    #mentions
    new_str = re.sub(r"@\w+", "", new_str)

    #hashtags
    new_str = re.sub(r"#(\w+)", r"\1", new_str)

    new_str = re.sub(' +', ' ', new_str)

    for char in punctuation:
        new_str = new_str.replace(char, ' ')
    
    new_str = re.sub(r"\s+", " ", new_str)
    
    return new_str.strip()


#Clean content:
new_data["clean_text"] = new_data["content"].apply(clean_text)

#Comparison:
print()
print(new_data[["content", "clean_text", "sentiment"]].sample(15))

print("BEFORE:", new_data["content"].iloc[0])
print("AFTER :", new_data["clean_text"].iloc[0])

