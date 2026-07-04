import json
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
    """
    Clean and preprocess input text.
    """
    text = text.lower()

    tokens = nltk.word_tokenize(text)

    processed = []

    for token in tokens:

        if token in string.punctuation:
            continue

        if token in stop_words:
            continue

        token = lemmatizer.lemmatize(token)

        processed.append(token)

    return " ".join(processed)


# Load intents dataset
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data["intents"]:

    tag = intent["tag"]

    for pattern in intent["patterns"]:

        sentences.append(preprocess(pattern))
        labels.append(tag)

print(f"Training Samples : {len(sentences)}")

# Convert text into vectors
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sentences)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X, labels)

# Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nTraining Completed Successfully!")

print("model.pkl created")
print("vectorizer.pkl created")
