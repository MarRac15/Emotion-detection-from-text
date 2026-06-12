import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import random
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = "resources/data/raw/tweet_emotions.csv"

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


def merge_classes(df: pd.DataFrame, merge_map):
    """
    Used for merging certain emotions that overlap with each other (due to simmilar semantic meaning)
    """
    df = df.copy()
    df["sentiment"] = df["sentiment"].replace(merge_map)
    return df



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
# print(data["sentiment"].value_counts())
# plt.figure(figsize=(10, 5))
# data['sentiment'].value_counts().plot(kind='bar')
# plt.title('Class Distribution')
# plt.xlabel('Emotion')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.show()


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

#merge fun to happiness and worry to sadness:
merge_map = {
    "fun": "happiness",
    "worry": "sadness"
}
new_data = merge_classes(new_data, merge_map)
print("Rozkład klas po zmergowaniu:")
print(new_data["sentiment"].value_counts())


#small resampling:
# df_major = new_data[new_data["sentiment"] == "sadness"]
# df_minor = new_data[new_data["sentiment"] == "hate"]

# df_major_downsampled = resample(
#     df_major,
#     replace=False,
#     n_samples=len(df_minor) * 5,
#     random_state=42
# )

# new_data = pd.concat([df_major_downsampled, df_minor])

#labels to id:
encoder = LabelEncoder()
new_data["label_id"] = encoder.fit_transform(new_data["sentiment"])
print("Mapowanie klas na id: ")
label_map = dict(enumerate(encoder.classes_))
print(label_map)

print()
print("\nRozkład klas po zbalansowaniu: ")
print(new_data["sentiment"].value_counts().sort_values(ascending=False))
print(round(new_data["sentiment"].value_counts(normalize=True) * 100, 2))





#Clean content:
new_data["clean_text"] = new_data["content"].apply(clean_text)

# print("Liczba powstałych pustych stringów:")
# print((new_data["clean_text"] == "").sum())
# print((new_data["clean_text"].str.strip() == "").sum())
# print()

#delete the empty strings from cleaning:
new_data = new_data[new_data["clean_text"].str.strip().astype(bool)]

#Comparison:
print()
print(new_data[["content", "clean_text", "sentiment"]].sample(15))
print()
print("BEFORE:", new_data["content"].iloc[0])
print("AFTER :", new_data["clean_text"].iloc[0])


#split:
train_df, temp_df = train_test_split(
    new_data,
    test_size=0.3,
    random_state=42,
    stratify=new_data["sentiment"]
)

#val + test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["sentiment"]
)

#check the stratify split
print(len(train_df), len(val_df), len(test_df))
print(train_df["sentiment"].value_counts(normalize=True))
print(val_df["sentiment"].value_counts(normalize=True))
print(test_df["sentiment"].value_counts(normalize=True))

print(f"liczba wszystkich wartości NaN w całej tabeli: {train_df.isna().sum().sum()}")
print(f"liczba wszystkich wartości null w całej tabeli: {train_df.isnull().sum()}")

print(f"liczba wszystkich wartości NaN w całej tabeli: {val_df.isna().sum().sum()}")
print(f"liczba wszystkich wartości null w całej tabeli: {val_df.isnull().sum()}")

print(f"liczba wszystkich wartości NaN w całej tabeli: {test_df.isna().sum().sum()}")
print(f"liczba wszystkich wartości null w całej tabeli: {test_df.isnull().sum()}")

train_df.to_csv("resources/data/train.csv", index=False)
val_df.to_csv("resources/data/val.csv", index=False)
test_df.to_csv("resources/data/test.csv", index=False)