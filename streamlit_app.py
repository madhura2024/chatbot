# app.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import streamlit as st

# sample data
data = {
    "text": [
        "hi", "hello", "hey", "good morning", "good evening",
        "i want to see kurtis", "show me sarees", "show me western wear", "i want kids clothing", "show me shoes",
        "are there any offers", "any discount", "do you have coupons", "what are today’s deals",
        "where is my order", "track my order", "order status", "i have not received my order",
        "cancel my order", "how to cancel my order", "i want to cancel",
        "how to return product", "return my item", "start a return",
        "how many days delivery", "when will i get my product", "delivery time estimate",
        "i need help with payment", "payment failed", "can i use upi", "what payment options are there",
        "what is your name", "who are you", "are you a human", "what can you do",
        "tell me a joke", "make me laugh", "say something funny",
        "i need help", "can you help me", "support"
    ],
    "label": [
        "greeting", "greeting", "greeting", "greeting", "greeting",
        "product_search", "product_search", "product_search", "product_search", "product_search",
        "offers", "offers", "offers", "offers",
        "order_tracking", "order_tracking", "order_tracking", "order_tracking",
        "order_cancel", "order_cancel", "order_cancel",
        "return_policy", "return_policy", "return_policy",
        "delivery_time", "delivery_time", "delivery_time",
        "payment_help", "payment_help", "payment_help", "payment_help",
        "question", "question", "question", "question",
        "joke", "joke", "joke",
        "help", "help", "help"
    ]
}
df = pd.DataFrame(data)

# vectorize
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['text'])
y = df['label']

# train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = MultinomialNB()
model.fit(x_train, y_train)

# response dict
responses = {
    "greeting": "Hello! How can I assist you today?",
    "product_search": "Sure! You can browse categories like Kurtis, Sarees, and more in the app.",
    "offers": "We have exciting offers running! Check the homepage for today’s deals.",
    "order_tracking": "You can track your order in the 'My Orders' section of the app.",
    "order_cancel": "To cancel an order, go to 'My Orders' and select the cancel option.",
    "return_policy": "You can return items within 7 days of delivery from the 'My Orders' section.",
    "delivery_time": "Delivery usually takes 3–7 business days depending on your location.",
    "payment_help": "You can pay via UPI, Cards, Wallets, and Cash on Delivery. Let me know what’s not working.",
    "question": "I’m Meesho Bot, your virtual shopping assistant! Ask me anything.",
    "joke": "Why did the shopping cart break up with the wishlist? It felt used.",
    "help": "I’m here to help! You can ask about orders, returns, payments, and more."
}

# Streamlit UI
st.title("🛍️ Meesho Chatbot")

user_input = st.text_input("Say something to Meesho Bot:")

if user_input:
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    response = responses.get(prediction[0], "Sorry, I didn’t understand that. Can you rephrase?")
    
    st.write("🤖 Bot:", response)
