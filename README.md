🤖 Simple Intent-Based Chatbot with PyTorch
This repository contains a basic yet functional chatbot built using Python, PyTorch, and a JSON-based intent classification system. The bot uses a neural network to recognize user intent based on their input and responds accordingly.

📁 What's Inside
intents.json — A structured dataset of sample user inputs (patterns) and corresponding responses for various intents.

chatbot.py — Core chatbot logic including:

Tokenization & lemmatization with NLTK

Bag-of-words vectorization

Feedforward neural network for classification

Intent-response mapping with optional function triggers

model_dimensions.json & chatbot_model.pth — Trained model and its dimensions (generated after training).

🧠 Intents Covered
Greetings

Goodbyes

Programming & coding questions

Learning resources (general)

Time & weather queries

Joke generation

Stock portfolio demo (calls a simple function)

🚀 How to Run
🔧 1. Install dependencies
bash
Copy
Edit
pip install torch nltk numpy
Also run:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('wordnet')
🧪 2. Train the model (First time)
python
Copy
Edit
# Uncomment these lines in main()
assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
assistant.parse_intents()
assistant.prepare_data()
assistant.train_model(batch_size=8, lr=0.001, epochs=100)
assistant.save_model('chatbot_model.pth', 'model_dimensions.json')
🔁 3. Use the trained chatbot
python
Copy
Edit
# Comment out training lines and run the following
assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
assistant.parse_intents()
assistant.load_model("chatbot_model.pth", "model_dimensions.json")

# Then interact in terminal:
Enter your message: Hello  
Assistant: Hi there! How can I help?
Type exit to quit the chat.

💡 Features
Simple NLP pre-processing using NLTK

Feedforward neural network with ReLU activations

Dynamic intent-response mapping

Extensible: plug in functions to intents (e.g., get_stocks())

📌 Note
This is an educational project demonstrating basic chatbot logic. It's ideal for beginners looking to understand how NLP, classification, and PyTorch tie together.
