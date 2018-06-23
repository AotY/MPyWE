#!/home/hefang/PROGRAMFILES/anaconda2/bin/python
# encoding: utf-8
from __future__ import division
import argparse
import math
import struct
import sys
import warnings
import os
import codecs
import numpy as np

from multiprocessing import Pool, Value, Array
import time
from collections import defaultdict
import pickle

u'''
chinese character enhanced word embedding.

adding tf-idf weights
质量也非常依赖于上下文窗口大小的选择

大的上下文窗口学到的词嵌入更反映主题信息，
而小的上下文窗口学到的词嵌入更反映词的功能和上下文语义信息

 TF-IDF classifies documents into categories inside the documents themselves.
'''

MIN_CHINESE = 0x4E00
MAX_CHINESE = 0x9FA5

character_size = (MAX_CHINESE - MIN_CHINESE + 1)

idf_characters_dict = pickle.load(codecs.open("./data/people's_daily_idf_characters_dict.pkl", "rb"))

tf_article_character_counter_dict = pickle.load(
    codecs.open("./data/people's_daily_tf_article_character_counter_dict.pkl", "rb"))

character_weight_dict = defaultdict()


# 词信息
class VocabItem:
    def __init__(self, word):
        self.word = word

        self.character = []  # morphme

        for character in word:
            character_ord = ord(character)
            if character_ord < MIN_CHINESE or character_ord > MAX_CHINESE:
                self.character = []
                break

            # 加入character的相对位置
            self.character.append(character_ord - MIN_CHINESE)

        self.count = 0
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


# 词汇表
class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}

        word_count = 0
        fi = codecs.open(fi, 'r', encoding='utf-8')

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in [u'<bol>', u'<eol>']:
            vocab_hash[token] = len(vocab_items)  # vocab index
            vocab_items.append(VocabItem(token))  #

        for line in fi:
            line = line.rstrip()
            if line == '' or line.startswith('#######'):
                continue

            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)  # vocab index
                    vocab_items.append(VocabItem(token))

                # assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1  # 出现次数加一
                word_count += 1

                if word_count % 10000 == 0:
                    sys.stdout.write(u"\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash[u'<bol>']].count += 1
            vocab_items[vocab_hash[u'<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell()
        self.vocab_items = vocab_items  # List of VocabItem objects
        self.vocab_hash = vocab_hash  # Mapping from each token to its index in vocab
        self.word_count = word_count  # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        # assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print(u'Total words in training file: %d' % self.word_count)
        print(u'Total bytes in training file: %d' % self.bytes)
        print(u'Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem(u'<unk>'))
        unk_hash = 0
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print()
        print(u'Unknown vocab size:', count_unk)

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash[u'<unk>'] for token in tokens]

    u'''
    构造霍夫曼树：https://www.wikiwand.com/zh-hans/%E9%9C%8D%E5%A4%AB%E6%9B%BC%E7%BC%96%E7%A0%81
    '''

    def encode_huffman(self):

        # Build a Huffman tree
        vocab_size = len(self)  # len 635
        count = [t.count for t in self] + [1e15] * (vocab_size - 1)  # len 1269

        parent = [0] * (2 * vocab_size - 2)  # len 1268
        binary = [0] * (2 * vocab_size - 2)  # len 1268

        # vocab 是按从大到小排序的
        pos1 = vocab_size - 1  # 634
        pos2 = vocab_size  # 635

        for i in range(vocab_size - 1):
            # Find min1  寻找最小频率1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]  # 合并最小出现次数的两个节点
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1  # 有点像桶标记思路

        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self):
            path = []  # List of indices from the leaf to the root
            code = []  # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size:
                    path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab):

        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant 正常化常量，用于当分母

        table_size = 1e8  # Length of the unigram table 100000000.0
        table = np.zeros(int(table_size), dtype=np.uint32)

        print(u'Filling unigram table')
        p = 0  # Cumulative probability 累积概率
        i = 0
        for j, token in enumerate(vocab):
            p += float(math.pow(token.count, power)) / norm  # p的最大值就是1
            while i < table_size and (float(i) / table_size) < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


# 这里是取近似值
def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


'''
初始化Matrix syn0, syn0_c, syn1
'''


def init_net(dim, vocab_size, character_size):  # dim=635, vocab_size=100

    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    #  用区间[-0.5,0.5] / dim的均匀分布的随机数初始化syn0
    tmp = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim))

    syn0_c = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(character_size, dim))

    # Create and return a ctypes object from a numpy array
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    syn0_c = np.ctypeslib.as_ctypes(syn0_c)
    syn0_c = Array(syn0_c._type_, syn0_c, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn0_c, syn1)


'''
根据pid来划分fi文件
'''


def train_process(pid):
    # Set fi to point to the right chunk of training file
    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)

    fi.seek(start)
    print(u'Worker %d beginning training at %d, ending at %d \n' % (pid, start, end))

    alpha = starting_alpha

    word_count = 0
    last_word_count = 0
    article_characters_tf_idf = None
    while fi.tell() < end:  #
        line = fi.readline().strip()

        # Skip blank lines
        if not line:
            continue

        if line.startswith(u'######'):
            file_name = line.split(' ')[-1]

            print(u'file_name: ', file_name)
            tf_article_character_counter = tf_article_character_counter_dict.get(file_name)

            if tf_article_character_counter is None or len(tf_article_character_counter) == 0:
                continue

            # 计算每一篇文章中每一个字（语素）的tf-idf
            article_characters_tf_idf = {}

            # 排序
            tf_article_character_counter = sorted(tf_article_character_counter.items(), key=lambda x: x[1],
                                                  reverse=True)
            max_count = tf_article_character_counter[0][1]
            for c, count in tf_article_character_counter:
                tf = count / max_count
                tf_idf = tf * idf_characters_dict[c]
                c_index = ord(c) - MIN_CHINESE
                article_characters_tf_idf[c_index] = tf_idf
                print(u'c: {}, tf_idf: {}, c_index: {}'.format(c, tf_idf, c_index))

        # Init sent, a list of indices of words in line
        sent = vocab.indices([u'<bol>'] + line.split() + [u'<eol>'])  # 构造一行，加上<bol> 和 <eol>

        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate alpha
                alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                if alpha < starting_alpha * 0.0001:
                    alpha = starting_alpha * 0.0001

                # Print progress info
                sys.stdout.write(u"\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                 (alpha, global_word_count.value, vocab.word_count,
                                  float(global_word_count.value) / vocab.word_count * 100))
                sys.stdout.flush()

            # Randomize window size, where win is the max window size 随机化窗口大小，其中win是最大窗口大小
            current_win = np.random.randint(low=1, high=win + 1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))

            # 前后上下文
            context = sent[context_start: sent_pos] + sent[sent_pos + 1: context_end]  # Turn into an iterator?

            # CBOW
            if cbow:

                neu1 = np.zeros(dim)
                neu1e = np.zeros(dim)

                character_index_list = []
                character_weight_list = []

                for c in context:
                    neu1c = np.zeros(dim)

                    neu1c += syn0[c]

                    if len(vocab[c].character) > 0:
                        # 获取 character 的 tf-idf ，标准化
                        characters_tf_idfs = [article_characters_tf_idf[character_index] for character_index in
                                              vocab[c].character]
                        norm_characters_tf_idfs = characters_tf_idfs / np.sum(characters_tf_idfs)
                        for character_index, weight in zip(vocab[c].character, norm_characters_tf_idfs):
                            neu1c += syn0_c[character_index] * weight
                            character_index_list.append(character_index)
                            character_weight_list.append(weight)

                            # 统计所有character的weight，然后求平均
                            # if character_index not in character_weight_dict:
                            #     character_weight_dict[character_index] = []

                            # character_weight_dict[character_index].append(weight)

                        neu1c = neu1c * 0.5

                    neu1 += neu1c

                assert len(neu1) == dim, u'neu1pinyin and dim do not agree'

                neu1 = np.multiply(neu1, 1.0 / len(context))

                # Compute neu1e and update syn1
                if neg > 0:
                    # negative sampling
                    classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                else:
                    # hierarchical softmax
                    classifiers = zip(vocab[token].path, vocab[token].code)  # 通过Huffman tree获取

                for target, label in classifiers:
                    z = np.dot(neu1, syn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)

                    neu1e += g * syn1[target]  # Error to backpropagate to syn0

                    syn1[target] += g * neu1  # Update syn1

                # Update syn0 # 哦，这里是这么更新的。
                for c in context:
                    syn0[c] += neu1e

                # character_rate: the factor <float> of learning rate for pinyin, default is 1.0
                for character_index, weight in zip(character_index_list, character_weight_list):
                    syn0_c[character_index] += neu1e * character_rate
                    # syn0_c[character_index] += neu1e * weight

            # Skip-gram
            else:
                for c in context:

                    # Error to backpropagate to syn0
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        # negative sampling
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        # hierarchical softmax
                        classifiers = zip(vocab[token].path, vocab[token].code)

                    neu1 = np.zeros(dim)
                    neu1c = np.zeros(dim)

                    neu1c += syn0[c]

                    if len(vocab[c].character) > 0:
                        # 获取 character 的 tf-idf ，标准化
                        characters_tf_idfs = [article_characters_tf_idf[character_index] for character_index in
                                              vocab[c].character]
                        norm_characters_tf_idfs = characters_tf_idfs / np.sum(characters_tf_idfs)
                        for character_index, weight in zip(vocab[c].character, norm_characters_tf_idfs):
                            neu1c += syn0_c[character_index] * weight

                        neu1c = neu1c * 0.5

                    neu1 += neu1c

                    for target, label in classifiers:
                        # z = np.dot(syn0[context_word], syn1[target])

                        z = np.dot(neu1, syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)

                        neu1e += g * syn1[target]  # Error to backpropagate to syn0
                        syn1[target] += g * syn0[c]  # Update syn1

                        # Update syn0
                    syn0[c] += neu1e

                    # Update syn0_c
                    if len(vocab[c].character) > 0:
                        for character_index in vocab[c].character:
                            syn0_c[character_index] += neu1e * character_rate
                            # for character_index, weight in zip(vocab[c].character, norm_characters_tf_idfs):
                            # syn0_c[character_index] += neu1e * weight

            word_count += 1

    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write(u"\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value) / vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()


u'''
保存 vector
'''


def save(vocab, syn0, syn0_c, fo, binary):
    print(u'Saving model to', fo)
    dim = len(syn0[0])
    if binary:
        fo = codecs.open(fo, 'wb', encoding='utf-8')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            for character_index in token.charater:
                vector = np.add(vector, np.multiply(syn0_c[character_index, :], 1.0 / len(token.charater)))

            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:  # 按字符串保存
        fo = codecs.open(fo, 'w', encoding='utf-8')
        fo.write('%d %d\n' % (len(syn0), dim))  # syn0, dim (635, 100)
        for token, vector in zip(vocab, syn0):
            word = token.word
            tmp_vector = np.zeros(dim)
            tmp_vector += vector

            for character_index in token.character:
                tmp_vector += syn0_c[character_index] * 1.0 / len(token.character)

            # 加上weight
            # if len(token.character) > 0:
            #     weights = []
            #     for character_index in token.character:
            #         character_weights = character_weight_dict.get(character_index)
            #         if character_weights is None or len(character_weights) == 0:
            #             weights.append(0)
            #         else:
            #             weights.append(np.mean(weights))
            #     print ('token.character: {}, weights: {}'.format(token.character, weights))
            #
            #     if len(weights) > 0 and np.sum(weights) > 0:
            #         norm_weights = weights / np.sum(weights)
            #
            #         for character_index, weight in zip(token.character, norm_weights):
            #             # tmp_vector = np.add(tmp_vector, np.multiply(syn0_c[character_index], 1.0 / len(token.character)))
            #             tmp_vector += np.multiply(syn0_c[character_index], weight)

            vector_str = ' '.join([str(s) for s in tmp_vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()


'''

'''


def __init_process(*args):
    global vocab, syn0, syn0_c, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, character_rate, global_word_count, fi

    # initargs = (vocab, syn0, syn1, table, cbow, neg, dim, alpha, win, num_processes, global_word_count, fi)

    vocab, syn0_tmp, syn0_m_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, character_rate, global_word_count = args[
                                                                                                                                          :-1]
    fi = codecs.open(args[-1], 'r', encoding='utf-8')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)
        syn0_c = np.ctypeslib.as_array(syn0_m_tmp)


'''

'''


def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary, character_rate):
    # Read train file to init vocab (词汇表）
    vocab = Vocab(fi, min_count)

    # Init net
    syn0, syn0_c, syn1 = init_net(dim, len(vocab), character_size)

    global_word_count = Value('i', 0)

    table = None

    #
    if neg > 0:
        print(u'Initializing unigram table')
        table = UnigramTable(vocab)
    else:
        print(u'Initializing Huffman tree')
        vocab.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()

    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn0_c, syn1, table, cbow, neg, dim, alpha,
                          win, num_processes, character_rate, global_word_count, fi))

    # Apply `func` to each element in `iterable`, collecting the results in a list that is returned.
    pool.map(train_process, range(num_processes))
    t1 = time.time()

    print()
    print(u'Completed training. Training took', (t1 - t0) / 60, u'minutes')

    # Save model to file
    save(vocab, syn0, syn0_c, fo, binary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train_file = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/input-chinese"
    # output_file = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/output-chinese"

    # /Users/LeonTao/NLP/Corpos/wiki/zhwiki-latest-simplified_tokened.txt
    train_file = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/people's_daily_cleaned"
    output_file = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/people's_daily_character_tfidf_skip_gram_100d"

    t0 = time.time()

    u'''
    修改内容：
    negative: 5
    '''
    parser.add_argument('-train', help='Training file', dest='fi', default=train_file)  # , required=True
    parser.add_argument('-model', help='Output model file', dest='fo', default=output_file)  # , required=True
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=1, type=int)
    parser.add_argument('-negative',
                        help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax',
                        dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5,
                        type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0,
                        type=int)
    parser.add_argument('-character-rate', help='the factor <float> of learning rate for character, default is 1.0',
                        dest='character_rate', default=1.0, type=float)
    # TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)

    print(u'os.getcwd: {}'.format(os.getcwd()))
    # -train data/input -model data/output -cbow 1 -negative 5 -dim 100 -window 5
    args = parser.parse_args()
    print(u'args: {} \n'.format(args))

    train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.win,
          args.min_count, args.num_processes, bool(args.binary), args.character_rate)

    t1 = time.time()
    print(u"cost time: {}".format(t1 - t0))
