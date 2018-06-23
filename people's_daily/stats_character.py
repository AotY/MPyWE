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
    total_character_counter = Counter()

    # 字在文章中出现的次数
    idf_articles_character_counter = Counter()

    # 文章中每一个字出现的次数
    tf_article_character_counter_dict = {}

    tf_article_character_counter = None
    article_token_characters = None
    last_file_name = None

    with codecs.open(cleaned_file_path, 'r', encoding='utf-8') as f:
        for line in f:

            # 某个词是否在某篇文章中出现
            line = line.rstrip()

            if line == '':
                continue

            # start a new article
            if line.startswith('###########'):
                file_name = line.split()[-1]
                if tf_article_character_counter is not None and article_token_characters is not None:
                    tf_article_character_counter.update(article_token_characters)
                    tf_article_character_counter_dict[last_file_name] = tf_article_character_counter

                    # 变为set
                    article_token_characters_set = set(article_token_characters)
                    idf_articles_character_counter.update(article_token_characters_set)

                article_count += 1

                last_file_name = file_name

                tf_article_character_counter = Counter()
                article_token_characters = []

                continue

            cleaned_words = line.split()

            cleaned_characters = []
            for character in cleaned_words:
                for c in character:
                    ord_c = ord(c)
                    if ord_c < MIN_CHINESE or ord_c > MAX_CHINESE:
                        continue
                        # break
                    cleaned_characters.append(c)

            total_character_counter.update(cleaned_characters)

            article_token_characters += cleaned_characters

        # 最后一篇文章
        # article_count += 1
        tf_article_character_counter.update(article_token_characters)
        tf_article_character_counter_dict[last_file_name] = tf_article_character_counter

        article_token_characters_set = set(article_token_characters)
        idf_articles_character_counter.update(article_token_characters_set)

        idf_characters_dict = {}
        # 计算每一个字的IDF（逆文档频率）
        for character, count in idf_articles_character_counter.iteritems():
            idf = math.log(article_count / (count + 1))
            idf_characters_dict[character] = idf
            print(u'character: {}, idf: {}'.format(character, idf))

        pickle.dump(idf_characters_dict, codecs.open("./../data/people's_daily_idf_characters_dict.pkl", "wb", encoding='utf-8'))

        pickle.dump(tf_article_character_counter_dict,
                    codecs.open("./../data/people's_daily_tf_article_character_counter_dict.pkl", "wb", encoding='utf-8'))

        pickle.dump(total_character_counter, codecs.open("./../data/people's_daily_total_character_counter.pkl", "wb", encoding='utf-8'))

        print('article_count: {}'.format(article_count))


if __name__ == '__main__':
    main()