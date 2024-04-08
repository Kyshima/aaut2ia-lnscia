import nltk
from nltk.chat.util import Chat, reflections
from utils import *
import yaml
import os


pairs = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you?', ['I am doing well, thank you!', 'I am great, thanks for asking!']),
    (r'(.*) your name?', ['My name is ChatBot.', 'I go by the name ChatBot.']),
    (r'(.*) (age|old) are you?', ['I am just a computer program, so I do not have an age.']),
    (r'what can you do?', ['I can have basic conversations with you.', 'I can answer your questions.', 'I am here to chat with you.']),
    (r'quit', ['Bye! Take care.', 'Goodbye!']),
]

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


intent_recognizer = unpickle_file(RESOURCE_PATH['INTENT_RECOGNIZER'])
tfidf_vectorizer = unpickle_file(RESOURCE_PATH['TFIDF_VECTORIZER'])

chatbot = Chat(pairs, reflections)

print("Welcome! Type 'quit' to exit.")
while True:
    user_input = input("You: ")

    prepared_question = text_prepare(user_input)

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

        response = 'Shit, that sounds serious af my brother!' #inserir tratamento dos modelos
        print("ChatBot:", response)
        if user_input.lower() == 'quit':
            break