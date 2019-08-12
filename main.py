import argparse
from pathlib import Path
import multiprocessing as mp
import pickle
import pandas as pd

import nltk
from nltk.corpus import stopwords

from src.glove import get_centroid, get_all_embeddings
from src.utils import parse_text, parse_all_text, jaccard_similarities, cosine_similarity, manhatten_similarity, \
    bag_of_words, tfidf

if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))

    with open(f"{Path(__file__).parent}/queries.txt") as f:
        text = f.readlines()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('query', type=str, help='a query to be compared against')
    parser.add_argument('path', default="../data/glove.42B.300d.txt", help='path to glove embeddings')

    args = parser.parse_args()

    input = args.query
    path = args.path


    cleaned_input = parse_text(input)

    indexes, all_words = parse_all_text(text + [input])

    bag_of_words_input = bag_of_words(cleaned_input, indexes)
    binary_bag_of_words_input = [1 if count > 0 else 0 for count in bag_of_words_input]

    jaccard_sims, cosine_sims, cosine_binary_sims, manhatten_sims, all_bag_of_words = [], [], [], [], []
    for line in text:
        cleaned_words = parse_text(line)

        bag_of_words_line = bag_of_words(cleaned_words, indexes)
        binary_bag_of_words_line = [1 if count > 0 else 0 for count in bag_of_words_line]
        all_bag_of_words.append(bag_of_words_line)

        jaccard_sim = jaccard_similarities(cleaned_input, cleaned_words)
        cosine_sim = cosine_similarity(bag_of_words_input, bag_of_words_line)
        cosine_binary_sim = cosine_similarity(binary_bag_of_words_input, binary_bag_of_words_line)
        manhatten_sim = manhatten_similarity(bag_of_words_input, bag_of_words_line)

        jaccard_sims.append((line, jaccard_sim))
        cosine_sims.append((line, cosine_sim))
        cosine_binary_sims.append((line, cosine_binary_sim))
        manhatten_sims.append((line, manhatten_sim))

    jaccard_sims.sort(key=lambda x: x[1], reverse=True)
    cosine_sims.sort(key=lambda x: x[1], reverse=True)
    cosine_binary_sims.sort(key=lambda x: x[1], reverse=True)
    manhatten_sims.sort(key=lambda x: x[1], reverse=False)

    print("\n--- Top 3 Jaccard Similarities --- ")
    for i in jaccard_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")

    print("\n--- Top 3 Cosine Similarities --- ")
    for i in cosine_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")
    print("\n--- Top 3 Cosine Similarities (Binary)--- ")
    for i in cosine_binary_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")

    print("\n--- Top 3 Manhatten Similarities --- ")
    for i in manhatten_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")

    # creating non lemmatized bag of words df
    indexes, all_words = parse_all_text(text + [input], lemmatization=False)
    cleaned_input = parse_text(input, lemmatization=False)  # Lemmatization not used for word embeddings
    bag_of_words_input = bag_of_words(cleaned_input, indexes)
    all_bag_of_words = [bag_of_words_input]
    for line in text:
        cleaned_words = parse_text(line, lemmatization=False)
        all_bag_of_words.append(bag_of_words(cleaned_words, indexes))

    bag_df = pd.DataFrame(all_bag_of_words, columns=indexes.keys())

    all_embeddings = get_all_embeddings(
        words=set(all_words).union(set(cleaned_input)),
        glove_path=path)

    input_embeddings, weighted_input_embeddings = [], []
    for word in cleaned_input:
        if word in all_embeddings:
            embedding = all_embeddings[word]
            weighted_embedding = all_embeddings[word] / tfidf(word, cleaned_input, bag_df)
            input_embeddings.append(embedding)
            weighted_input_embeddings.append(weighted_embedding)

    input_centroid = get_centroid(input_embeddings)
    weighted_input_centroid = get_centroid(weighted_input_embeddings)

    glove_sims, weighted_glove_sims = [], []
    for line in text:
        cleaned_words = parse_text(line, lemmatization=False)

        query_embeddings, weighted_query_embeddings = [], []
        for word in cleaned_words:
            if word in all_embeddings:
                embedding = all_embeddings[word]
                weighted_embedding = all_embeddings[word] / tfidf(word, cleaned_words, bag_df)
                query_embeddings.append(embedding)
                weighted_query_embeddings.append(weighted_embedding)

        query_centroid = get_centroid(query_embeddings)
        weighted_query_centroid = get_centroid(weighted_query_embeddings)

        glove_sims.append((line, cosine_similarity(input_centroid, query_centroid)))
        weighted_glove_sims.append((line, cosine_similarity(weighted_input_centroid, weighted_query_centroid)))

    glove_sims.sort(key=lambda x: x[1], reverse=True)
    weighted_glove_sims.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Top 3 Glove Similarities --- ")
    for i in glove_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")

    print("\n--- Top 3 Glove Similarities Weighted with TFIDF --- ")
    for i in weighted_glove_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")
