import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences

if __name__ == "__main__":
    df = pd.read_csv("output.csv")
    # Assuming 'descriptions' is a column in your DataFrame containing the textual descriptions
    X = df['visual_description']
    y = df['crop']  # Replace 'labels' with the actual column name for your categories

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()

    # Convert TF-IDF matrices to sequences
    X_train_sequences = [np.where(row > 0)[0] for row in X_train_tfidf]
    X_test_sequences = [np.where(row > 0)[0] for row in X_test_tfidf]

    # Pad sequences to have a consistent length
    max_sequence_length = max(len(seq) for seq in X_train_sequences + X_test_sequences)
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Build LSTM model
    embedding_dim = 100
    num_classes = len(np.unique(y))
    model = Sequential()
    model.add(Embedding(input_dim=len(vectorizer.get_feature_names_out()), output_dim=embedding_dim))
    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_padded, y_train, epochs=200, batch_size=50, validation_split=0.3)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test_padded).argmax(axis=1)

    # Calculate and print additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')