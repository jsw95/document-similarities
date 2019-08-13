import numpy as np

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
