import pickle
import matplotlib.pyplot as plt

# Load the training history from the file
with open('C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/training_history3_2.pkl', 'rb') as file:
    saved_history = pickle.load(file)

# Plot the graph of accuracy versus epochs
plt.plot(saved_history['accuracy'], label='Training Accuracy')
plt.plot(saved_history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.xlim(0, 10)  # Set the x-axis limits from 0 to 10
plt.show()

