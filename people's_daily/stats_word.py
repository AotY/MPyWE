# encoding: utf-8
from __future__ import division

from collections import Counter
import os
from os.path import join, getsize
from util import parse_line
import pickle as pkl
import math
from remove_stop_words import load_stop_words, remove_words
import codecs

'''
人民日报
dataset_folder_path = "/Users/LeonTao/NLP/Corpos/People's Daily 2014"

统计每个词的TF - IDF
'''

# dataset_folder_path = "/Users/LeonTao/NLP/Corpos/People's Daily 2014"

cleaned_file_path = "./../data/people's_daily_cleaned"


# cleaned_file_path = "./../data/input-chinese"

MIN_CHINESE = 0x4E00
MAX_CHINESE = 0x9FA5

def main():
    # 遍历所有文件
    article_count = 0

    # 字在语料库中的总次数
    total_word_counter = Counter()

    # 字在文章中出现的次数
    idf_articles_word_counter = Counter()

    # 文章中每一个字出现的次数
    tf_article_word_counter_dict = {}

    tf_article_word_counter = None
    article_token_words = None
    last_file_name = None

    with codecs.open(cleaned_file_path, 'r', encoding='utf-8') as f:
        for line in f:

            # 某个词是否在某篇文章中出现
            line = line.rstrip()

            if line == '':
                continue

            # start a new article
            if line.startswith(u'###########'):
                file_name = line.split()[-1]
                if tf_article_word_counter is not None and article_token_words is not None:
                    tf_article_word_counter.update(article_token_words)
                    tf_article_word_counter_dict[last_file_name] = tf_article_word_counter

                    # 变为set
                    article_token_words_set = set(article_token_words)
                    idf_articles_word_counter.update(article_token_words_set)

                article_count += 1
                # print ('article_count: ', article_count)

                last_file_name = file_name

                tf_article_word_counter = Counter()
                article_token_words = []

                continue

            tmp_words = line.split()
            cleaned_words = []
            for word in tmp_words:
                is_all_chinese = True
                for c in word:
                    ord_c = ord(c)
                    if ord_c < MIN_CHINESE or ord_c > MAX_CHINESE:
                        is_all_chinese = False
                        break
                if is_all_chinese:
                    cleaned_words.append(word)

            total_word_counter.update(cleaned_words)

            article_token_words += cleaned_words

        # 最后一篇文章
        # article_count += 1
        tf_article_word_counter.update(article_token_words)
        tf_article_word_counter_dict[last_file_name] = tf_article_word_counter

        article_token_words_set = set(article_token_words)
        idf_articles_word_counter.update(article_token_words_set)

        idf_words_dict = {}
        # 计算每一个字的IDF（逆文档频率）
        for word, count in idf_articles_word_counter.iteritems():
            idf = math.log(article_count / (count + 1))
            idf_words_dict[word] = idf
            print(u'word: {}, idf: {}'.format(word, idf))

        pkl.dump(idf_words_dict, codecs.open("./../data/people's_daily_idf_words_dict.pkl", "wb"))

        pkl.dump(tf_article_word_counter_dict,
                 codecs.open("./../data/people's_daily_tf_article_word_counter_dict.pkl", "wb"))

        pkl.dump(total_word_counter, codecs.open("./../data/people's_daily_total_word_counter.pkl", "wb"))

        print('article_count: {}'.format(article_count))
        # print(u'idf_words_dict: {}'.format(idf_words_dict))


if __name__ == '__main__':
    main()
