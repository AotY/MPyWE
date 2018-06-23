# encoding: utf-8

# 将词性标注记号删除
def parse_line(line):
    # 中括号问题 [壳牌/nz 石油/n 公司/nis]/nt 修改为  [ 壳牌/nz 石油/n 公司/nis ]/nt
    line = line.replace('[', '[ ')
    line = line.replace(']', ' ]')
    new_tokens = []
    tokens = line.split()
    for token in tokens:
        if token:
            new_tokens.append(token.split('/')[0])

    return ' '.join(new_tokens)