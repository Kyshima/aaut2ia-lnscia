import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.layers import Dropout
from keras.src.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from utils import *
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv("new.csv", sep=";")
    # Assuming 'descriptions' is a column in your DataFrame containing the textual descriptions
    X_text = df['visual_description'].apply(text_prepare)
    y = df.apply(lambda row: row['crop'] + re.compile(' ').sub('', row['disease']), axis=1).apply(text_prepare)  # Replace 'labels' with the actual column name for your categories
    df['formality'].values.reshape(-1, 1)

    # One-hot encode the new categorical feature
    df['formality'] = df['formality'].fillna("informal")
    formality = df['formality'].values.reshape(-1, 1)
    formality_encoder = LabelEncoder()
    formality_encoded = formality_encoder.fit_transform(formality)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X_text).toarray()

    # Convert TF-IDF matrices to sequences
    X_sequences = [np.where(row > 0)[0] for row in X_tfidf]

    # Pad sequences to have a consistent length
    max_sequence_length = max(len(seq) for seq in X_sequences)
    X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

    X = np.hstack((X_padded, formality_encoded.reshape(-1, 1)))
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Build LSTM model
    num_classes = len(np.unique(y))
    epochs = 40
    batch_size = 45
    learning_rate = 0.01
    embedding_dim = 100
    max_sequence_length = 100
    dropout_rate = 0.2

    model = Sequential()
    model.add(Embedding(input_dim=len(vectorizer.get_feature_names_out()), output_dim=embedding_dim))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=125, return_sequences=True))
    model.add(Dropout(0.15))

    model.add(LSTM(units=200))
    model.add(Dropout(0.1))

    model.add(Dense(units=num_classes, activation='softmax'))


    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3, callbacks=[early_stopping])

    # Evaluate the model on the test set
    y_pred = model.predict(X_test).argmax(axis=1)

    # Calculate and print additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


    form = formality_encoder.transform(["formal"])

    X_tfidf = vectorizer.transform([text_prepare("The wheat leaves display small, rusty-brown spots scattered across their surface, accompanied by a gradual yellowing of the surrounding tissue.")]).toarray()

    # Convert TF-IDF matrices to sequences
    X_sequences = [np.where(row > 0)[0] for row in X_tfidf]

    X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)
    X = np.hstack((X_padded, [form]))

    print(model.predict(X))
    print(label_encoder.inverse_transform(model.predict(X).argmax(axis=1)))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot the graph of accuracy versus epochs
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()

    # Plot the graph of loss versus epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()