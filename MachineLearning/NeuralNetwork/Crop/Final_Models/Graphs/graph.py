import pickle
import matplotlib.pyplot as plt

# Load the training history
with open('C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/Final_Models/Graphs/training_history1_crop.pkl', 'rb') as file:
#with open('C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/Final_Models/Graphs/training_history2_crop.pkl', 'rb') as file:
#with open('C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/NeuralNetwork/Crop/Final_Models/Graphs/training_history3_crop.pkl', 'rb') as file:
    history = pickle.load(file)

# Plot the graph of accuracy versus epochs
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

# Plot the graph of loss versus epochs
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
