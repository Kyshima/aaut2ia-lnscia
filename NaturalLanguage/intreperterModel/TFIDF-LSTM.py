import pandas as pd
import numpy as np
from keras.src.layers import Dropout
from keras.src.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences

if __name__ == "__main__":
    df = pd.read_csv("new.csv", sep=";")
    # Assuming 'descriptions' is a column in your DataFrame containing the textual descriptions
    X_text = df['visual_description']
    y = df['crop'].astype(str) + df['disease']  # Replace 'labels' with the actual column name for your categories
    df['formality'].values.reshape(-1, 1)

    # One-hot encode the new categorical feature
    df['formality'] = df['formality'].fillna("informal")
    formality = df['formality'].values.reshape(-1, 1)
    label_encoder = LabelEncoder()
    formality_encoded = label_encoder.fit_transform(formality)

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
    epochs = 20
    batch_size = 64
    learning_rate = 0.001
    embedding_dim = 100
    max_sequence_length = 100
    dropout_rate = 0.2

    model = Sequential()
    model.add(Embedding(input_dim=len(vectorizer.get_feature_names_out()), output_dim=embedding_dim))
    model.add(LSTM(128))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

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