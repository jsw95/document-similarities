import re
import time

import numpy as np


def timeit(func):
    def run(*args, **kwargs):
        print(f"\nRunning func: {func.__name__}")
        t0 = time.time()
        a = func(*args, **kwargs)
        t1 = time.time()

        print(f"\nTime taken for {func.__name__}: {round(t1 - t0, 3)}s\n")
        return a

    return run


@timeit
def get_all_embeddings(words, glove_path):
    regex = ""
    for word in words:
        regex += f"(^{word} )|"
    regex = regex[:-1]  # stripping final |
    embeds = {}

    with open(glove_path) as f:
        for line in f:
            if re.match(regex, line[:20]):
                line = line.split()
                embeds[line[0]] = np.array([float(i) for i in line[1:]])

    return embeds


def get_centroid(query_embeds):
    centroid = np.mean(query_embeds, axis=0)

    return centroid
