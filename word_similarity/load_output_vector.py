# encoding: utf-8
import gensim
import logging

'''
使用gensim加载训练后的output文件
'''

def load_vec(path=None, binary=False):
    if path is None:
        path = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/people's_daily_cbow_200d"
    # Load pre-trained vec

    # model = gensim.models.Word2Vec.load_word2vec_format(path, binary=binary)
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
    # word_similarity = model.wv.word_similarity(u'学校', u'学生')
    # print word_similarity
    return model



if __name__ == '__main__':
    model = load_vec()
    file_path = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/240.txt"
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model.accuracy(file_path)