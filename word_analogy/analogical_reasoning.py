# encoding: utf-8
from word_similarity.load_output_vector import load_vec
import logging

'''
Analogical reasoning
'''


def ar(model, file_path=None):
    if file_path is None:
        file_path = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/word_analogy.txt"
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model.accuracy(file_path)


if __name__ == '__main__':
    model = load_vec()
    ar(model)
    pass
