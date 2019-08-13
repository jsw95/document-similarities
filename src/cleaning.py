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