import importlib
from utils import *
# from dialogue_manager import DialogueManager
import dialogue_manager
importlib.reload(dialogue_manager)

dialogue_mngr = dialogue_manager.DialogueManager(RESOURCE_PATH)

questions = [
    "Hey",
    "How are you doing?",
    "What's your hobby?",
    "Help me",
    "How to write a loop in python?",
    "How to delete rows in pandas?",
    "python3 re",
    "What is the difference between c and c++",
    "Multithreading in Java",
    "Catch exceptions C++",
    "What is AI?",
]

for question in questions:
    answer = dialogue_mngr.generate_answer(question)
    print('Q: %s\nA: %s \n' % (question, answer))