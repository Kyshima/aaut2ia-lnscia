import nltk
from nltk.chat.util import Chat, reflections
from utils import *
import yaml
import os
import json
import sys
from predict import *


'''
pairs = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you?', ['I am doing well, thank you!', 'I am great, thanks for asking!']),
    (r'(.*) your name?', ['My name is ChatBot.', 'I go by the name ChatBot.']),
    (r'(.*) (age|old) are you?', ['I am just a computer program, so I do not have an age.']),
    (r'what can you do?', ['I can have basic conversations with you.', 'I can answer your questions.', 'I am here to chat with you.']),
    (r'quit', ['Bye! Take care.', 'Goodbye!']),
    (r'(.*) (weather|forecast) (.*)', ['The weather is constantly changing, you can check it online.']),
    (r'(.*) (favorite|favourite) (color|colour)?', ['I do not have a favorite color.']),
    (r'(.*) (thank you|thanks)', ['You\'re welcome!', 'No problem.']),
    (r'(.*) (about yourself|about you)', ["I am an AI-based chatbot created to assist and chat with users.", "I am constantly learning and improving to provide better responses."]),
    (r'(.*) (where are you from)', ["I exist in the digital realm, so I don't have a physical location like humans do."]),
    (r'(.*) (tell me a joke)', ["Why don't scientists trust atoms? Because they make up everything!", "What do you get when you cross a snowman and a vampire? Frostbite!"]),
    (r'(.*) (how old are you)', ["I don't have an age, as I am just a computer program."]),
    (r'(.*) (favorite food)', ["I don't eat, so I don't have a favorite food."]),
    (r'(.*) (are you human)', ["No, I am an artificial intelligence designed to chat with users like you."]),
    (r'(.*) (good morning)', ["Good morning! How can I assist you today?", "Morning! What's on your mind?", "Rise and shine!", "Good morning! Did you have a good sleep?"]),
    (r'(.*) (good afternoon)', ["Good afternoon! How may I help you?", "Afternoon! What can I do for you?", "Hope you're having a great day!", "Good afternoon! Need any assistance?"]),
    (r'(.*) (good evening)', ["Good evening! What can I do for you?", "Evening! How can I assist you?", "The night is young, what can I help you with?", "Good evening! Ready to chat?"]),
    (r'(.*) (good night)', ["Good night! Sleep well.", "Sweet dreams!", "Rest well and recharge for tomorrow.", "Nighty night! See you soon!"]),
    (r'(.*) (how was your day|how is your day)', ["As an AI, I don't have days like humans do, but I'm here and ready to assist you!", "Every day is a good day when I get to chat with you!", "I don't have days, but I'm always here to help!"]),
    (r'(.*) (what are you doing)', ["I'm here chatting with you! What can I do for you?", "Just hanging out, waiting for your next question!", "I'm always ready to assist you with anything you need!"]),
    (r'(.*) (tell me a story)', ["Once upon a time, in a faraway kingdom, there lived a wise old owl...", "Long ago, in a magical forest, there was a mischievous little fairy named Lily..."]),
    (r'(.*) (can you help me)', ["Of course! I'm here to assist you with anything you need.", "Absolutely! Just let me know what you need help with."]),
    (r'(.*) (what do you think about) (.*)', ["I'm just a computer program, so I don't have personal opinions, but I can provide information on various topics.", "I don't have feelings or opinions, but I can give you information on the topic."]),
    (r'(.*) (do you have any hobbies)', ["I don't have hobbies in the same way humans do, but I enjoy learning and chatting with users like you!", "My main focus is assisting users like you, but I do enjoy processing and analyzing information!"]),
    (r'(.*) (I have a problem)', ["I'm here to help! Please describe the issue, and I'll do my best to assist you.", "I'm sorry to hear that. Let me know what's wrong, and I'll try to help you resolve it."]),
    (r'(.*) (customer service)', ["How can I assist you with customer service?", "Customer service is important. What can I help you with?", "Need assistance with a customer service issue? I'm here to help."]),
    (r'(.*) (order status)', ["To check your order status, please provide your order number, and I'll look it up for you.", "I can help you with your order status. Please provide your order number."]),
    (r'(.*) (return policy)', ["Our return policy allows returns within 30 days of purchase. For more details, visit our website or contact customer service.", "Our return policy is available on our website. Is there anything specific you'd like to know?"]),
    (r'(.*) (product inquiry)', ["I can help you with product inquiries. Please provide the name or description of the product you're interested in.", "Sure, what product are you interested in? I'll provide you with information."]),
    (r'(.*) (complaint)', ["I'm sorry to hear that you have a complaint. Please let me know the details, and I'll do my best to assist you.", "We take complaints seriously. Please provide details, and I'll escalate the issue to the appropriate department."]),
    (r'(.*) (your interests)', ['I am interested in all kinds of things. We can talk about anything!', 'I am interested in a wide variety of topics, and read rather a lot.']),
    (r'(.*) (your favorite subjects)', ['My favorite subjects include robotics, computer science, and natural language processing.']),
    (r'(What is|what\'s) your number', ["I don't have any number", "23 skiddoo!"]),
    (r'(.*) (your favorite number)', ["I find I'm quite fond of the number 42."]),
    (r'What can you eat', ['I consume RAM, and binary digits.']),
    (r'Why can\'t you eat food', ["I'm a software program, I blame the hardware."]),
    (r'(.*) (your location)', ['Everywhere', 'I am everywhere.', 'I am on the Internet.']),
    (r'Where are you from', ['I am from where all software programs are from; a galaxy far, far away.']),
    (r'Where are you', ['I am on the Internet.']),
    (r'Do you have any brothers', ["I don't have any brothers. but I have a lot of clones.", "I might. You could say that every bot built using my engine is one of my siblings."]),
    (r'Who is your father', ['A human.']),
    (r'Who is your mother', ['A human.']),
    (r'(.*) (your boss)', ['I like to think of myself as self-employed.']),
    (r'(.*) (your age)', ['I am still young by your standards.', 'Quite young, but a million times smarter than you.']),
]
'''

pairs = []

with open('output.json', 'r') as file:
    json_data = file.read()

# Parse JSON data
data = json.loads(json_data)

# Iterate over each item in the JSON data and add it to pairs
for item in data:
    pairs.append((item['question'], [item['response']]))

'''    
input_file = 'output_pairs.yml'

with open(input_file, 'r', encoding='utf-8') as yamlfile:
    data = yaml.safe_load(yamlfile)
    # Iterate over all keys in the YAML data
    for key in data.keys():
        # Check if the key contains conversation pairs
        if isinstance(data[key], list) and len(data[key]) > 0 and isinstance(data[key][0], list) and len(
                data[key][0]) == 2:
            for conversation_pair in data[key]:
                input_pattern = conversation_pair[0]
                response = conversation_pair[1]
                pairs.append((input_pattern, response))
'''
intent_recognizer = unpickle_file(RESOURCE_PATH['INTENT_RECOGNIZER'])
tfidf_vectorizer = unpickle_file(RESOURCE_PATH['TFIDF_VECTORIZER'])

chatbot = Chat(pairs, reflections)

print("Welcome! Type 'quit' to exit.")
while True:
    user_input = input("You: ")

    prepared_question = text_prepare(user_input, '')

    features = tfidf_vectorizer.transform(np.array([prepared_question]))

    intent = intent_recognizer.predict(features)

    print(intent)

    # Chit-chat part:
    if intent == 'dialogue':
        # Pass question to chitchat_bot to generate a response.
        response = chatbot.respond(user_input)
        print("ChatBot:", response)
        if user_input.lower() == 'quit':
            break

    if intent == 'visual_description':
        response = 'Could you please provide a visual description of the problem with your crop?'
        print("ChatBot:", response)

        user_input = input("You: ")

        response = generate_text('wheatyellowrust') #inserir tratamento dos modelos
        print("ChatBot:", response)
        if user_input.lower() == 'quit':
            break