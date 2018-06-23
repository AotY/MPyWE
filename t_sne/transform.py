#!/home/hefang/PROGRAMFILES/anaconda2/bin/python
# encoding: utf-8
# from __future__ import unicode_literals
import numpy as np
import sys
from sklearn.manifold import TSNE
import os
import codecs
import logging
from collections import Counter
import pandas as pd

vector_dict = {}

def read_vector(path):
    assert os.path.isfile(path), "{} is not a file.".format(path)
    embedding_dim = -1
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break

    with codecs.open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        all_lines = len(lines)
        index = 0
        for index, line in enumerate(lines):
            values = line.strip().split(' ')
            if len(values) == 1 or len(values) == 2:
                continue
            if len(values) != int(embedding_dim) + 1:
                # print("Warning {} -line.".format(index + 1))
                logging.info("Warning {} -line.".format(index + 1))
                continue
            vector_dict[values[0]] = np.array(list(map(float, values[1:])))
            if index % 2000 == 0:
                sys.stdout.write("\rHandling with the {} lines, all {} lines.".format(index + 1, all_lines))
        sys.stdout.write("\rHandling with the {} lines, all {} lines.".format(index + 1, all_lines))
    print("\nembedding words {}, embedding dim {}.".format(len(vector_dict), embedding_dim))



'''
统计词出现次数
'''
def stats_meta_data(path):
    word_counter = Counter()
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip blank lines
            if not line:
                continue

            if line.startswith(u'######'):
                continue

            words = line.split()
            word_counter.update(words)
    # sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    return word_counter



if __name__ == '__main__':

    meta_path = "./../data/people's_daily_cleaned"
    word_counter = stats_meta_data(meta_path)

    vec_path = "./../data/people's_daily_character_pinyin_word_tfidf_cbow_100d"
    read_vector(vec_path)

    vector_list = []
    word_list = []
    for word in word_counter.keys():
        if vector_dict.get(word) is not None:
            vector_list.append(vector_dict.get(word))
            word_list.append(word)

    # pop words which aren't in word_counter
    for word in word_counter.keys():
        if word not in word_list:
            word_counter.pop(word)

    df = pd.DataFrame(
        data={
            'word': word_counter.keys(),
            'count': word_counter.values()
        }
    )
    df.sort_values(by='count', inplace=True, ascending=False)
    df.to_csv('word_meta_data.tsv', sep='\t', index=False, header=True, encoding='utf-8')
    print (df.shape)
    del df
    import time
    t0 = time.time()
    X = np.array(vector_list)
    # X_embedded = TSNE(n_components=3, perplexity=30.0).fit_transform(X)
    tsne = TSNE(n_components=4, perplexity=35.0, method='exact', learning_rate=100)
    X_embedded = tsne.fit_transform(X)
    del tsne
    X_embedded_df = pd.DataFrame(data=X_embedded, columns=['x1', 'x2', 'x3', 'x4'])
    print (X_embedded_df.shape)
    X_embedded_df.to_csv('vec_data.tsv', sep='\t', index=False, header=False, encoding='utf-8')
    print ('time cost: ', time.time() - t0)