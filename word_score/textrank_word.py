# encoding: utf-8
from __future__ import division
# from __future__ import unicode_literals
from collections import Counter
import os
import pickle as pkl
import math
import codecs
from textrank import KeywordTextRank

'''
人民日报
dataset_folder_path = "/Users/LeonTao/NLP/Corpos/People's Daily 2014"

统计每个词的 textrank
'''

cleaned_file_path = "./../data/people's_daily_cleaned"

MIN_CHINESE = 0x4E00
MAX_CHINESE = 0x9FA5


def main():
    textrank_article_word_counter_dict = {}
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
                if article_token_words is not None:
                    if len(article_token_words) > 0:
                        keyword_rank = KeywordTextRank(article_token_words)
                        keyword_rank.solve()

                        vertex_dict = keyword_rank.get_vertex_dict()
                        textrank_article_word_counter_dict[last_file_name] = vertex_dict
                        keyword_rank = None

                last_file_name = file_name

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

            article_token_words.append(cleaned_words)

        # 最后一篇文章
        if len(article_token_words) > 0:
            keyword_rank = KeywordTextRank(article_token_words)
            keyword_rank.solve()

            vertex_dict = keyword_rank.get_vertex_dict()
            textrank_article_word_counter_dict[last_file_name] = vertex_dict
            keyword_rank = None

        pkl.dump(textrank_article_word_counter_dict,
                 codecs.open("./../data/people's_daily_textrank_article_word_counter_dict.pkl", "wb"))

        print(u'textrank_article_word_counter_dict: {}'.format(textrank_article_word_counter_dict))


if __name__ == '__main__':
    main()
