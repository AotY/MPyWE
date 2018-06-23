# encoding: utf-8
import codecs

import os

'''
去除停顿词
'''


def load_stop_words():
    stop_words = []
    file_path = './../data/stop_words.txt'
    with codecs.open(file_path, 'r', encoding='utf-8') as fi:
        for line in fi:
            line = line.strip()
            # Skip blank lines
            if not line:
                continue

            if line not in stop_words:
                stop_words.append(line)

    return stop_words



def clean(stop_words):
    file_path = "./../data/people's_daily"
    save_file_f = codecs.open("./../data/people's_daily_cleaned", 'w', encoding='utf-8')

    with codecs.open(file_path, 'r', encoding='utf-8') as fi:
        for line in fi:
            line = line.strip()
            # Skip blank lines
            if not line:
                continue

            if line.startswith('########'):
                save_file_f.write(line + os.linesep)
                continue

            cleaned_words = remove_words(line, stop_words)
            save_file_f.write(' '.join(cleaned_words) + os.linesep)

    save_file_f.close()


def remove_words(line, stop_words=None):
    cleaned_words = [token for token in line.split() if token not in stop_words]
    return cleaned_words


if __name__ == '__main__':
    stop_words = load_stop_words()
    clean(stop_words)
