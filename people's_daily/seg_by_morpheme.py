# encoding: utf-8
'''
将分词后的词尝试用语素划分，

如果，计算字的tf-idf呢 ？
这个不知道效果咋样 ？
'''
from pybloom import BloomFilter
import codecs

monosyllableMorphemePath = './../dict/all_monosyllable_morpheme.txt'
disyllableMorphemePath = './../dict/all_disyllable_morpheme.txt'
multisyllableMorphemePath = './../dict/all_multisyllable_morpheme.txt'

class MorphemeSeg:
    def __init__(self):
        self.monosyllableMorphemeBf = BloomFilter(capacity=20000, error_rate=0.0001)
        self.disyllableMorphemeBf = BloomFilter(capacity=50000, error_rate=0.0001)
        self.multisyllableMorphemeBf = BloomFilter(capacity=500000, error_rate=0.0001)

        # 加载单音节，双音节，多音节语素表
        self.load_morphemes()


    def load_morphemes(self):
        print('load morpheme list')

        pathList = [monosyllableMorphemePath, disyllableMorphemePath, multisyllableMorphemePath]
        for i, path in enumerate(pathList):
            with codecs.open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()
                    if i == 0:
                        if line not in self.monosyllableMorphemeBf:
                            self.monosyllableMorphemeBf.add(line)
                    elif i == 1:
                        if line not in self.disyllableMorphemeBf:
                            self.disyllableMorphemeBf.add(line)
                    else:
                        if line not in self.multisyllableMorphemeBf:
                            self.multisyllableMorphemeBf.add(line)

    # 判断curWord是否在词典中，多音节，双音节和单音节语素表中
    def isInDict(self, curWord):
        wordLen = len(curWord)
        if wordLen == 1:
            return curWord in self.monosyllableMorphemeBf
        elif wordLen == 2:
            return curWord in self.disyllableMorphemeBf
        else:
            return curWord in self.multisyllableMorphemeBf

    # 正向最大匹配
    def MM(self, sentence):
        words = []

        if len(sentence) == 0:
            return words

        tmpSentence = sentence
        maxLen = len(tmpSentence)

        frontIndex = 0
        rearIndex = maxLen
        while True:
            if frontIndex >= maxLen:
                return words

            for i in range(rearIndex, frontIndex - 1, -1):

                # 如果字典都不能匹配，则将当前字认为单独一个成分，跳过
                if i <= frontIndex:
                    if tmpSentence[frontIndex] != ' ' and tmpSentence[frontIndex] != '\t':  # 舍弃空格
                        words.append(tmpSentence[frontIndex])
                    frontIndex += 1
                    break

                curWord = tmpSentence[frontIndex:i]

                if self.isInDict(curWord):
                    words.append(curWord.strip().replace(' ', ''))
                    frontIndex = i
                    break

    # 逆向最大匹配
    def RMM(self, sentence):
        words = []
        if len(sentence) == 0:
            return words

        tmpSentence = sentence
        maxLen = len(tmpSentence)

        frontIndex = 0
        rearIndex = maxLen
        while True:
            if rearIndex <= 0:
                return words

            for i in range(frontIndex, rearIndex + 1):
                # 如果字典都不能匹配，则将当前字认为单独一个成分，跳过
                if i > rearIndex - 1:
                    rearIndex -= 1
                    if tmpSentence[rearIndex] != ' ' and tmpSentence[frontIndex] != '\t':  # 舍弃空格
                        words.append(tmpSentence[rearIndex])
                    break

                curWord = tmpSentence[i: rearIndex]

                if self.isInDict(curWord):
                    words.append(curWord.strip().replace(' ', ''))
                    rearIndex = i
                    break


    # 双向匹配算法，切分句子
    def cut(self, sentence):
        print('cut sentence: {}'.format(sentence))

        # 正向最大匹配思想MM 算法
        words1 = self.MM(sentence)
        # print('words1: {}'.format('/'.join(words1)))

        # 逆向最大匹配RMM 算法
        words2 = self.RMM(sentence)
        # print('words2: {}'.format('/'.join(words2)))

        words2.reverse()

        # 启发式比较
        if len(words1) == len(words2):
            # 长度相同
            # a.分词结果相同，就说明没有歧义，可返回任意一个。
            if words1 == words2:
                return words1

            # b.分词结果不同，返回其中单字较少的那个。
            monoCount1 = 0
            monoCount2 = 0
            for i in range(len(words1)):
                if len(words1[i]) == 1:
                    monoCount1 += 1
                if len(words2[i]) == 1:
                    monoCount2 += 1
            return words1 if monoCount1 <= monoCount2 else words2
        else:
            # 如果长度不一致，返回较小长度的一个结果
            # print('words1: {}/'.format(' '.join(words1)))
            # print('words2: {}/'.format(' '.join(words2)))
            return words1 if len(words1) >= len(words2) else words2


if __name__ == '__main__':

    ms = MorphemeSeg()
    for token in ms.cut('环境保护'):
        print(token)
    pass