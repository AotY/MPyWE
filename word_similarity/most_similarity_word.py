# encoding: utf-8
import gensim
import logging
import sys

def load_model(fname, binary=False):
    # Load Google's pre-trained Word2Vec model.
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.KeyedVectors.load_word2vec_format(fname=fname, binary=binary)
    return model

if __name__ == '__main__':
    print("Word Similarity Evaluation")

    # vector_file = "./Data/zhwiki_substoke.100d.source"
    vector_file = vector_file = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/people's_daily_character_word_tfidf_cbow_100d_2"

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
    model = load_model(vector_file)
    while True:
        word = raw_input('input word: ').decode('utf-8')

        # word = word
        # print (type(word))
        print (u'word: {}'.format(word))
        print (u'most similarity words: ')
        similarity_words = model.wv.most_similar(word, topn=20)
        for item in similarity_words:
            print (u'word: {}, sim: {}'.format(item[0], item[1]))
