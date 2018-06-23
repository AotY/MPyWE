# encoding: utf-8
import codecs
import os
from os.path import join, getsize
from util import parse_line

dataset_folder_path = "/Users/LeonTao/NLP/Corpos/People's Daily 2014"

save_file_path = "/Users/LeonTao/PycharmProjects/deborausujono/word2vecpy/data/people's_daily"


def main():
    # 遍历所有文件
    count = 0
    size = 0
    save_file = open(save_file_path, 'w')
    for root, dirs, file_names in os.walk(dataset_folder_path):
        if len(dirs) != 0:
            continue
        print root, dirs, file_names
        for file_name in file_names:
            if file_name == '.DS_Store':  # avoid error in  macos
                continue
            file_path = join(root, file_name)
            # open file
            size += getsize(file_path)
            if size < 5:
                continue

            save_file.write('###################### %s' % file_name + os.linesep)
            with open(file_path, 'r') as f:
                count += 1
                for line in f:
                    line = line.rstrip()

                    if line == '':
                        continue

                    # print line
                    new_line = parse_line(line)
                    # save line
                    save_file.write(new_line + os.linesep)
            # new article separate

    save_file.close()

    print 'count: {}'.format(count)
    print 'size: {}'.format(size)
    print 'size (m): {}'.format(size / (1024 * 1024))

    '''
    count: 52846
    size: 186289409
    size (m): 177
    '''


if __name__ == '__main__':
    main()
    pass
