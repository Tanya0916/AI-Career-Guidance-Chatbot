import json
import random
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Load Model
# -----------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Load Vectorizer
# -----------------------------
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# -----------------------------
# Load Intents
# -----------------------------
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)


# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess(text):

    text = text.lower()

    words = text.split()

    processed = []

    for word in words:

        word = word.strip(string.punctuation)

        if word == "":
            continue

        if word in stop_words:
            continue

        word = lemmatizer.lemmatize(word)

        processed.append(word)

    return " ".join(processed)


# -----------------------------
# Greeting Detection
# -----------------------------
greetings = [
    "hi",
    "hello",
    "hey",
    "hii",
    "helo",
    "good morning",
    "good afternoon",
    "good evening",
    "what's up",
    "sup"
]

greeting_responses = [
    "Hello! 👋 How can I help you today?",
    "Hi! 😊 Ask me anything about your career.",
    "Welcome! What career guidance do you need today?",
    "Hello! I'm your AI Career Guidance Assistant."
]


# -----------------------------
# Thanks Detection
# -----------------------------
thanks_words = [
    "thanks",
    "thank you",
    "thankyou",
    "thx"
]

thanks_responses = [
    "You're welcome! 😊",
    "Happy to help!",
    "Anytime! Feel free to ask more questions."
]


# -----------------------------
# Goodbye Detection
# -----------------------------
bye_words = [
    "bye",
    "goodbye",
    "see you",
    "exit",
    "quit"
]

bye_responses = [
    "Goodbye! 👋",
    "See you again.",
    "Best of luck with your career!"
]


# -----------------------------
# Main Chat Function
# -----------------------------
def get_response(user_input):

    user = user_input.lower().strip()

    # Greeting
    if user in greetings:
        return random.choice(greeting_responses)

    # Thanks
    if user in thanks_words:
        return random.choice(thanks_responses)

    # Goodbye
    if user in bye_words:
        return random.choice(bye_responses)

    # NLP Processing
    processed = preprocess(user)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector).max()

    print("-----------------------------------")
    print("User :", user)
    print("Processed :", processed)
    print("Intent :", prediction)
    print("Confidence :", probability)
    print("-----------------------------------")

    # Lower confidence threshold
    if probability < 0.15:
        return (
            "❌ Sorry, I couldn't understand your question.\n\n"
            "You can ask me about:\n\n"
            "• AI Career\n"
            "• Machine Learning\n"
            "• Data Science\n"
            "• Resume Tips\n"
            "• Interview Preparation\n"
            "• Web Development\n"
            "• Certifications\n"
            "• Internships\n"
            "• Cloud Computing\n"
            "• Cyber Security"
        )

    # Find response
    for intent in intents["intents"]:

        if intent["tag"] == prediction:

            return random.choice(intent["responses"])

    return "Sorry, I couldn't find an answer."