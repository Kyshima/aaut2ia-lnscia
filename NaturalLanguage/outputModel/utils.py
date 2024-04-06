import nltk
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'data/word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    #text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine previously.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    embeddings = {}
    for line in open(embeddings_path, encoding='utf-8'):
        w, *v = line.strip().split('\t')
        embeddings[w] = np.asfarray(v, dtype=np.float32)

    embeddings_dim = next(iter(embeddings.values())).size

    return embeddings, embeddings_dim
    

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    # Hint: you have already implemented exactly this function previously.

    list_question = question.split()
    sum_vector = np.zeros(dim)
    n_vectors = 0
    for w in list_question:
        if w in embeddings:
            sum_vector += embeddings[w]
            n_vectors += 1
    qv_emb = sum_vector
    if n_vectors > 1:
        qv_emb = sum_vector / n_vectors

    return qv_emb


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def tfidf_features(X_train, X_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""

    # Train a vectorizer on X_train data.
    # Transform X_train and X_test data.

    # Pickle the trained vectorizer to 'vectorizer_path'
    # Don't forget to open the file in writing bytes mode.

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=9000, ngram_range=(1, 2), token_pattern=r'(\S+)')

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    return X_train, X_test