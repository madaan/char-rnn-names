"""
generate ngrams
"""
from collections import Counter
from src.features.char_codec import CharCodec
def get_ngrams(filename, n):
    ngrams = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().replace(" ", "_")
            name_len = len(line)
            for i in range(name_len - n):
                ngrams.append(line[i : i + n])
    print(Counter(ngrams))

import os
print(os.getcwd())
n = 3
get_ngrams("../../data/indian_names.txt", n)
get_ngrams("../../data/caucasian_names.txt", n)
get_ngrams("../../data/hispanic_names.txt", n)
get_ngrams("../../data/african_american_names.txt", n)