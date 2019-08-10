import sys
from pathlib import Path
from pprint import pprint

import nltk
from nltk.corpus import stopwords

from src.glove import get_centroid, get_all_embeddings
from src.utils import parse_text, parse_all_text, jaccard_similarities, cosine_similarity, manhatten_similarity, \
    bag_of_words

if __name__ == "__main__":
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    with open(f"{Path(__file__).parent}/queries.txt") as f:
        text = f.readlines()

    input = sys.argv[1]

    # input = "Can you advise me as to how I go about petitioning for a divorce without a marriage certificate. Having " \
    #         "been married overseas I am unable to obtain a copy of certificate and have not seen my partner/wife for " \
    #         "30 years. Would greatly appreciate any advice."

    cleaned_input = parse_text(input)

    indexes, all_words = parse_all_text(text)

    all_embeddings = get_all_embeddings(set(all_words).union(set(cleaned_input)))
    input_embeddings = [all_embeddings[word] for word in cleaned_input if word in all_embeddings]

    input_centroid = get_centroid(input_embeddings)

    jaccard_sims, cosine_sims, manhatten_sims, glove_sims = [], [], [], []
    for line in text:
        cleaned_words = parse_text(line)

        bag_of_words_input = bag_of_words(cleaned_input, indexes)
        bag_of_words_line = bag_of_words(cleaned_words, indexes)

        jaccard_sim = jaccard_similarities(cleaned_input, cleaned_words)
        cosine_sim = cosine_similarity(bag_of_words_input, bag_of_words_line)
        manhatten_sim = manhatten_similarity(bag_of_words_input, bag_of_words_line)

        query_embeddings = [all_embeddings[word] for word in cleaned_words if word in all_embeddings]
        line_centroid = get_centroid(query_embeddings)

        jaccard_sims.append((line, jaccard_sim))
        cosine_sims.append((line, cosine_sim))
        manhatten_sims.append((line, manhatten_sim))
        glove_sims.append((line, cosine_similarity(input_centroid, line_centroid)))  # TODO check logic

    jaccard_sims.sort(key=lambda x: x[1], reverse=True)
    cosine_sims.sort(key=lambda x: x[1], reverse=True)
    manhatten_sims.sort(key=lambda x: x[1], reverse=False)
    glove_sims.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Top 3 Jaccard Similarities --- ")
    for i in jaccard_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")


    print("\n--- Top 3 Cosine Similarities --- ")
    for i in cosine_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")


    print("\n--- Top 3 Manhatten Similarities --- ")
    for i in manhatten_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")

    print("\n--- Top 3 Glove Similarities --- ")
    for i in glove_sims[:3]:
        print(f"Query: {i[0]}Score: {round(i[1], 3)}\n")

