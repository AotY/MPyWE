# encoding: utf-8
from __future__ import division

from collections import Counter
import os
from os.path import join, getsize
from util import parse_line
import pickle
import math
from remove_stop_words import load_stop_words, remove_words
import codecs

'''
人民日报
dataset_folder_path = "/Users/LeonTao/NLP/Corpos/People's Daily 2014"

统计每个词的 IDF
'''

dataset_folder_path = "/Users/LeonTao/NLP/Corpos/People's Daily 2014"


stop_words = load_stop_words()

def main():

    # 遍历所有文件
    count = 0
    size = 0
    article_count = 0

    # 词在语料库中的总次数
    total_word_counter = Counter()

    # 词在文章中出现的次数
    articles_counter = Counter()

    # 文章中每一个词出现的次数
    article_word_counter_dict = {}

    for root, dirs, file_names in os.walk(dataset_folder_path):
        if len(dirs) != 0:
            continue
        print root, dirs, file_names

        for file_name in file_names:
            if file_name == '.DS_Store':  # avoid error in  macos
                continue
            file_path = join(root, file_name)
            # open file
            size += getsize(file_path)

            if size < 5:
                continue

            article_word_counter = Counter()
            article_token_words = []
            with codecs.open(file_path, 'r', encoding='utf-8') as f:
                count += 1
                for line in f:

                    # 某个词是否在某篇文章中出现
                    line = line.rstrip()

                    if line == '':
                        continue

                    # print line
                    new_line = parse_line(line)

                    # token_words = new_line.split()
                    cleaned_words = remove_words(new_line, stop_words)
                    total_word_counter.update(cleaned_words)

                    article_token_words += cleaned_words

            article_word_counter.update(article_token_words)
            article_word_counter_dict[file_name] = article_word_counter

            # 变为set
            article_token_words_set = set(article_token_words)
            articles_counter.update(article_token_words_set)

            article_count += 1

        words_idf = {}
        # 计算每一个词的IDF（逆文档频率）
        for word, count in articles_counter.iteritems():
            idf = math.log(article_count / (count + 1))
            words_idf[word] = idf

        pickle.dump(words_idf, open("./../data/people's_daily_words_idf.pkl", "wb"))

        pickle.dump(article_word_counter_dict, open("./../data/people's_daily_article_word_counter_dict.pkl", "wb"))

        pickle.dump(total_word_counter, open("./../data/people's_daily_word_counter.pkl", "wb"))


        print('words_idf: {}'.format(words_idf))

if __name__ == '__main__':
    main()