# encoding: utf-8
import codecs

import os


def clean():
    file_path = "./../data/people's_daily_cleaned"
    save_file_f = codecs.open("./../data/people's_daily_cleaned_without_filename", 'w', encoding='utf-8')

    with codecs.open(file_path, 'r', encoding='utf-8') as fi:
        for line in fi:
            line = line.strip()
            # Skip blank lines
            if not line:
                continue

            if line.startswith('########'):
                continue
            print (u'line: {}'.format(line))
            save_file_f.write(line + os.linesep)
            # cleaned_words = line.split()
            # print (u'cleaned_words: ', u' '.join(cleaned_words))
            # save_file_f.write(' '.join(cleaned_words) + os.linesep)

    save_file_f.close()

if __name__ == '__main__':
    clean()
