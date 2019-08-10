import numpy as np
import re
from pathlib import Path
# from main import cosine_similarity
import time

def timeit(func):
    def run(*args):
        print(f"\nRunning func: {func.__name__}")
        t0 = time.time()
        a = func(*args)
        t1 = time.time()

        print(f"\nTime taken for {func.__name__}: {round(t1 -t0, 3)}s\n")
        return a
    return run



@timeit
def get_all_embeddings(words):
    regex = ""
    for word in words:
        regex += f"(^{word} )|"
    regex = regex[:-1]  # stripping final |
    embeds = {}
    with open("../data/glove.42B.300d.txt") as f:
        for line in f:
            if re.match(regex, line[:20]):
                line = line.split()
                embeds[line[0]] = np.array([float(i) for i in line[1:]])

    return embeds


def get_embeddings(cleaned_words):
    regex = ""
    for word in cleaned_words:
        regex += f"(^{word} )|"
    regex = regex[:-1]  # stripping final |

    embeds = {}
    with open("short.txt") as f:

        for line in f:
            if re.match(regex, line[:20]):
                line = line.split()
                embeds[line[0]] = np.array([float(i) for i in line[1:]])

    return embeds


def get_centroid(query_embeds):
    centroid = np.mean(query_embeds, axis=0)

    return centroid


# centroid = get_centroid(["some", "mother", "footballer"])
# centroid2 = get_centroid(["footballers", "wife", "ball"])
# centroid3 = get_centroid(["the", "it", "dog"])
#
# print(cosine_similarity(centroid, centroid2))
# print(cosine_similarity(centroid, centroid3))