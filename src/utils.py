from nltk.corpus import stopwords
import string
from collections import Counter
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def parse_text(words, lemmatization=True):

    """Takes in list of strings, returns list of lowercase, lemmatized strings w/o punctuation and stopwords"""
    cleaned_words = []
    regex_slash = re.compile(f'/')
    words = regex_slash.sub(' ', words)  # catch for this/that words

    words = words.split()
    stop_words = set(stopwords.words('english'))
    regex = re.compile(f'[{re.escape(string.punctuation + "“")}]')

    for word in words:
        if word not in stop_words:
            word = regex.sub('', word)
            word = word.lower()
            if lemmatization:
                word = lemmatizer.lemmatize(word)
            cleaned_words.append(word)

    return cleaned_words


def parse_all_text(text, lemmatization=True):
    all_words = []
    parsed_text = []
    for line in text:
        words = parse_text(line, lemmatization)
        parsed_text.extend(words)
        all_words.extend(words)

    all_words = list(set(all_words))
    indexes = {word: index for index, word in enumerate(all_words)}

    return indexes, all_words


def bag_of_words(words, indexes):
    bow = [0] * len(indexes)

    counts = Counter(words)
    for word in words:
        if word in indexes:
            bow[indexes[word]] = counts[word]

    return np.array(bow)


def jaccard_similarities(input_query, query):
    """
    Jaccard Coefficient = (the number in both sets) / (the number in either set)
    Both query input are a list of parsed strings
    """

    s1 = set(input_query)
    s2 = set(query)

    jaccard = len(s1.intersection(s2)) / (len(s1) + len(s2) - len(s1.intersection(s2)))

    return jaccard


def cosine_similarity(input_query, query):
    similarity = np.dot(input_query, query) / (np.linalg.norm(input_query) * np.linalg.norm(query))
    return similarity


def manhatten_similarity(input_query, query):
    similarity = np.sum(abs(input_query - query))
    return similarity

def tfidf(word, doc_words, bag_df):
    norm_term_freq = doc_words.count(word) / len(doc_words)
    if np.count_nonzero(bag_df[word]) == 0:
        print(9)
    inv_doc_freq = np.log(len(bag_df) / np.count_nonzero(bag_df[word]))

    tfidf = norm_term_freq * inv_doc_freq
    return tfidf
