# encoding: utf-8
from load_output_vector import load_vec
import os

'''
compute semantic relatedness of given word pairs.
'''


def compute_semantic(model, file_path=None):
    if file_path is None:
        file_path = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/240.txt"
    test_score_file = open(file_path.split('.')[0] + 'test_score' + '.' + file_path.split('.')[1], 'w')

    with open(file_path, 'r') as f:
        for line in f:
            try:
                line = line.rstrip()
                word1, word2, score = line.split()
                print word1, word2, score
                sim = model.wv.similarity(word1.decode('utf-8'), word2.decode('utf-8'))
                test_score_file.write(' '.join([word1, word2, str(sim)]) + os.linesep)
            except:
                continue


if __name__ == '__main__':
    model = load_vec()
    compute_semantic(model)
    pass
