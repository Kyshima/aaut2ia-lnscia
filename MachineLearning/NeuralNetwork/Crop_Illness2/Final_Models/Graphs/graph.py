import pickle
import matplotlib.pyplot as plt

# Load the training history from the file
with open('C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Illness/Graphs/training_history3_illness.pkl', 'rb') as file:
    saved_history = pickle.load(file)

# Plot the graph of accuracy versus epochs
plt.plot(saved_history['accuracy'], label='Training Accuracy')
plt.plot(saved_history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.xlim(0, 10)
plt.ylim(0, 1)
plt.show()

