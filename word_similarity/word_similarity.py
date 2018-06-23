# encoding: utf-8
# @Author : bamtercelboo
# @Datetime : 2018/4/23 19:00
# @File : word_similarity.py
# @Last Modify Time : 2018/4/23 19:00
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  word_similarity.py
    FUNCTION : None
"""

import os
import sys
import logging
import numpy as np
from scipy import linalg, stats
import codecs

class Similarity(object):
    def __init__(self, vector_file, similarity_file):
        self.vector_file = vector_file
        self.similarity_file = similarity_file
        self.vector_dict = {}
        self.result = {}
        self.read_vector(self.vector_file)
        if self.similarity_file is "":
            self.Word_Similarity(similarity_name="./Data/wordsim-240.txt", vec=self.vector_dict)
            self.Word_Similarity(similarity_name="./Data/wordsim-297.txt", vec=self.vector_dict)
        else:
            self.Word_Similarity(similarity_name=self.similarity_file, vec=self.vector_dict)
        self.pprint(self.result)

    def read_vector(self, path):
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
                self.vector_dict[values[0]] = np.array(list(map(float, values[1:])))
                if index % 2000 == 0:
                    sys.stdout.write("\rHandling with the {} lines, all {} lines.".format(index + 1, all_lines))
            sys.stdout.write("\rHandling with the {} lines, all {} lines.".format(index + 1, all_lines))
        print("\nembedding words {}, embedding dim {}.".format(len(self.vector_dict), embedding_dim))

    def pprint(self, result):
        from prettytable import PrettyTable
        x = PrettyTable(["Dataset", "Found", "Not Found", "Score (rho)"])
        x.align["Dataset"] = "l"
        for k, v in result.items():
            x.add_row([k, v[0], v[1], v[2]])
        print(x)
        print ('self.vector_file: {}'.format(self.vector_file))

    def cos(self, vec1, vec2):
        return vec1.dot(vec2) / (linalg.norm(vec1) * linalg.norm(vec2))

    def rho(self, vec1, vec2):
        return stats.stats.spearmanr(vec1, vec2)[0]

    def Word_Similarity(self, similarity_name, vec):
        pred, label, found, notfound = [], [], 0, 0
        with codecs.open(similarity_name, 'r', encoding='utf-8') as fr:
            for i, line in enumerate(fr):
                w1, w2, score = line.split()
                if w1 in vec and w2 in vec:
                    # if np.any(np.isnan(vec[w1]), np.isnan(vec[2])):
                    #     continue
                    # vec[w1] = np.nan_to_num(vec[w1], copy=True)
                    # vec[w2] = np.nan_to_num(vec[w2], copy=True)
                    found += 1
                    pred.append(self.cos(vec[w1], vec[w2]))
                    label.append(float(score))
                else:
                    notfound += 1
        file_name = similarity_name[similarity_name.rfind("/") + 1:].replace(".txt", "")
        self.result[file_name] = (found, notfound, self.rho(label, pred))


if __name__ == "__main__":
    print("Word Similarity Evaluation")

    # vector_file = "./Data/zhwiki_substoke.100d.source"
    vector_file = vector_file = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/people's_daily_morpheme_pinyin_word_text_rank_cbow_100d"
    # similarity_file = "./Data/wordsim-297.txt"
    # Similarity(vector_file=vector_file, similarity_file=similarity_file)
    Similarity(vector_file=vector_file, similarity_file="")


    # parser = OptionParser()
    # parser.add_option("--vector", dest="vector", help="vector file")
    # parser.add_option("--similarity", dest="similarity", default="", help="similarity file")
    # (options, args) = parser.parse_args()
    #
    # vector_file = options.vector
    # similarity_file = options.similarity

    # try:
    #     Similarity(vector_file=vector_file, similarity_file=similarity_file)
    #     print("All Finished.")
    # except Exception as err:
    #     print(err)
















'''
zhwiki_cbow_200d：
embedding words 649720, embedding dim 200.
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  293  |     4     |  0.5389647037514111 |
| wordsim-240 |  239  |     1     | 0.46772839576662967 |
+-------------+-------+-----------+---------------------+


people's_daily_cbow_100d:
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    | 0.43360117369245027 |
| wordsim-240 |  233  |     7     |  0.328047129942329  |
+-------------+-------+-----------+---------------------+

+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  271  |     26    | 0.4449572863199858 |
| wordsim-240 |  233  |     7     | 0.3631522927361488 |
+-------------+-------+-----------+--------------------+


people's_daily_character_cbow_100d
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.4387167611028872 |
| wordsim-240 |  233  |     7     | 0.45791545494134034 |
+-------------+-------+-----------+---------------------+


people's_daily_pinyin_cbow_100d: 
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    | 0.24854081192661703 |
| wordsim-240 |  233  |     7     | 0.29411128537842607 |
+-------------+-------+-----------+---------------------+


people's_daily_morpheme_cbow_100d： 
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    |  0.3303406180821376 |
| wordsim-240 |  233  |     7     | 0.35662925217647884 |
+-------------+-------+-----------+---------------------+

people's_daily_pinyin_cbow_100d:
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    | 0.34220816676676086 |
| wordsim-240 |  233  |     7     |  0.3766589856207014 |
+-------------+-------+-----------+---------------------+

people's_daily_morpheme_cbow_100d: 
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    |  0.3303406180821376 |
| wordsim-240 |  233  |     7     | 0.35662925217647884 |
+-------------+-------+-----------+---------------------+


people's_daily_character_word_tfidf_cbow_100d, syn0[c] += neu1e * word_weight
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.474657965708301  |
| wordsim-240 |  233  |     7     | 0.46041015862131307 |
+-------------+-------+-----------+---------------------+



people's_daily_cbow_100d:
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    | 0.43360117369245027 |
| wordsim-240 |  233  |     7     |  0.328047129942329  |
+-------------+-------+-----------+---------------------+


people's_daily_pinyin_cbow_100d：(有提升么)
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  273  |     24    | 0.3753082597026683 |
| wordsim-240 |  233  |     7     | 0.3016660057709281 |
+-------------+-------+-----------+--------------------+



people's_daily_character_cbow_100d
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    | 0.44216233352306544 |
| wordsim-240 |  233  |     7     |  0.3567502161312978 |
+-------------+-------+-----------+---------------------+


people's_daily_morpheme_cbow_100d： 
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    |  0.3303406180821376 |
| wordsim-240 |  233  |     7     | 0.35662925217647884 |
+-------------+-------+-----------+---------------------+


people's_daily_pinyin_cbow_100d:
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    | 0.34220816676676086 |
| wordsim-240 |  233  |     7     |  0.3766589856207014 |
+-------------+-------+-----------+---------------------+

people's_daily_morpheme_cbow_100d: 
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    |  0.3303406180821376 |
| wordsim-240 |  233  |     7     | 0.35662925217647884 |
+-------------+-------+-----------+---------------------+


people's_daily_cbow_100d: 
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  271  |     26    | 0.4449572863199858 |
| wordsim-240 |  233  |     7     | 0.3631522927361488 |
+-------------+-------+-----------+--------------------+


people's_daily_character_word_tfidf_cbow_100d, syn0[c] += neu1e * word_weight
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.474657965708301  |
| wordsim-240 |  233  |     7     | 0.46041015862131307 |
+-------------+-------+-----------+---------------------+


people's_daily_character_word_tfidf_cbow_100d_2, syn0[c] += neu1e 
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.5403260779964207 |
| wordsim-240 |  233  |     7     | 0.43827043431028717 |
+-------------+-------+-----------+---------------------+




people's_daily_pinyin_word_tfidf_cbow_100d:
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  273  |     24    | 0.5346866953011037 |
| wordsim-240 |  233  |     7     | 0.420646123512216  |
+-------------+-------+-----------+--------------------+

+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  271  |     26    |  0.5306644184719   |
| wordsim-240 |  233  |     7     | 0.4580530217919188 |
+-------------+-------+-----------+--------------------+



people's_daily_word_tfidf_cbow_100d:
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.2990339544928927 |
| wordsim-240 |  233  |     7     | 0.18812784160046878 |
+-------------+-------+-----------+---------------------+

people's_daily_word_tfidf_cbow_100d_2
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  271  |     26    | 0.4017692341218852 |
| wordsim-240 |  233  |     7     | 0.3359524004878336 |
+-------------+-------+-----------+--------------------+

chaacter_word_tfidf_cbow_100d_2, syn0[c] += neu1e 
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.5403260779964207 |
| wordsim-240 |  233  |     7     | 0.43827043431028717 |
+-------------+-------+-----------+---------------------+



people's_daily_pinyin_word_tfidf_cbow_100d:
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  273  |     24    | 0.5346866953011037 |
| wordsim-240 |  233  |     7     | 0.420646123512216  |
+-------------+-------+-----------+--------------------+

+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.5478684533283076 |
| wordsim-240 |  233  |     7     | 0.46362863859701087 |
+-------------+-------+-----------+---------------------+




people's_daily_word_tfidf_cbow_100d:
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    | 0.41267779279123284 |
| wordsim-240 |  233  |     7     | 0.41159138068097695 |
+-------------+-------+-----------+---------------------+


people's_daily_morpheme_cbow_100d
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  273  |     24    | 0.4345643078596092 |
| wordsim-240 |  233  |     7     | 0.3541321766542548 |
+-------------+-------+-----------+--------------------+


people's_daily_cbow_100d
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  271  |     26    | 0.4449572863199858 |
| wordsim-240 |  233  |     7     | 0.3631522927361488 |
+-------------+-------+-----------+--------------------+


people's_daily_pinyin_cbow_100d
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  271  |     26    | 0.3816552046002204 |
| wordsim-240 |  233  |     7     | 0.4449082720349226 |
+-------------+-------+-----------+--------------------+


people's_daily_pinyin_skip_grim_100d
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.4523850187015741 |
| wordsim-240 |  233  |     7     | 0.43802518582149735 |
+-------------+-------+-----------+---------------------+


people's_daily_pinyin_word_tfidf_cbow_100d
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.5367093675442454 |
| wordsim-240 |  233  |     7     | 0.41748825348772106 |
+-------------+-------+-----------+---------------------+



people's_daily_character_pinyin_word_tfidf_cbow_100d
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.5450330838348831 |
| wordsim-240 |  233  |     7     | 0.48482258217758495 |
+-------------+-------+-----------+---------------------+


people's_daily_character_pinyin_cbow_100d
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  271  |     26    | 0.4425889681984607 |
| wordsim-240 |  233  |     7     | 0.4938759040509979 |
+-------------+-------+-----------+--------------------+
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    | 0.40295481026051555 |
| wordsim-240 |  233  |     7     | 0.39196240027046936 |
+-------------+-------+-----------+---------------------+
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  273  |     24    | 0.4139548901640408 |
| wordsim-240 |  233  |     7     | 0.3744393268356392 |
+-------------+-------+-----------+--------------------+
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  273  |     24    |  0.4029336964908448 |
| wordsim-240 |  233  |     7     | 0.39199436888104683 |
+-------------+-------+-----------+---------------------+
+-------------+-------+-----------+--------------------+
| Dataset     | Found | Not Found |    Score (rho)     |
+-------------+-------+-----------+--------------------+
| wordsim-297 |  273  |     24    | 0.4139548901640408 |
| wordsim-240 |  233  |     7     | 0.3744393268356392 |
+-------------+-------+-----------+--------------------+



people's_daily_morpheme_pinyin_word_tfidf_cbow_100d
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    |  0.5367093675442454 |
| wordsim-240 |  233  |     7     | 0.41750912569458276 |
+-------------+-------+-----------+---------------------+

people's_daily_morpheme_pinyin_word_tfidf_cbow_100d_2
+-------------+-------+-----------+---------------------+
| Dataset     | Found | Not Found |     Score (rho)     |
+-------------+-------+-----------+---------------------+
| wordsim-297 |  271  |     26    | 0.36044645990458957 |
| wordsim-240 |  233  |     7     |  0.4523173255830287 |
+-------------+-------+-----------+---------------------+

people's_daily_morpheme_pinyin_word_text_rank_cbow_100d
+-------------+-------+-----------+----------------------+
| Dataset     | Found | Not Found |     Score (rho)      |
+-------------+-------+-----------+----------------------+
| wordsim-297 |  271  |     26    | 0.041390968059511514 |
| wordsim-240 |  233  |     7     | 0.05805415968100717  |
+-------------+-------+-----------+----------------------+

'''