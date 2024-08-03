import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def preprocess(text):
    return word_tokenize(text.lower())

user_input = "Hello! How can I book a ticket?"
tokens = preprocess(user_input)
print(tokens)
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Create a new ChatBot instance
chatbot = ChatBot('MyBot')

trainer = ListTrainer(chatbot)

# Train the chatbot with a few responses
trainer.train([
    "Hi there!",
    "Hello",
    "How can I help you?",
    "I want to book a ticket.",
    "Sure, I can help with that. Where do you want to go?"
])

# Get a response to an input statement
response = chatbot.get_response("I want to book a ticket.")
print(response)
import spacy

nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

user_input = "Book a flight from New York to San Francisco"
entities = extract_entities(user_input)
print(entities)
from flask import Flask, request, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

app = Flask(__name__)
chatbot = ChatBot('MyBot')
trainer = ListTrainer(chatbot)
trainer.train(["Hi", "Hello", "How are you?", "I'm good, thanks!"])

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = chatbot.get_response(user_input)
    return jsonify({"response": str(response)})

if __name__ == "__main__":
    app.run(debug=True)
