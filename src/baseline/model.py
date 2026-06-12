from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

TRAIN_PATH = "resources/data/train.csv"
TEST_PATH = "resources/data/test.csv"
VAL_PATH = "resources/data/val.csv"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["clean_text"]
y_train = train_df["label_id"]

X_val = val_df["clean_text"]
y_val = val_df["label_id"]

X_test = test_df["clean_text"]
y_test = test_df["label_id"]


#TF_IDF vectors representation:
vectorizer = TfidfVectorizer(analyzer="char", max_features=20000, ngram_range=(3,5))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)


baseline_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced")


baseline_model.fit(X_train_vec, y_train)

y_pred = baseline_model.predict(X_test_vec)

print("Class report:")
print(classification_report(y_test, y_pred))
print()
cm = confusion_matrix(y_test, y_pred)
# print(confusion_matrix(y_test, y_pred))
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()

#error analysis:
test_df = test_df.copy()
test_df["pred"] = y_pred
wrong = test_df[test_df["label_id"] != test_df["pred"]]
print(wrong[["clean_text", "label_id", "pred"]].head(20))


# text = ["i am so happy today"]
# text_vec = vectorizer.transform(text)
# print(baseline_model.predict(text_vec))