#!/home/hefang/PROGRAMFILES/anaconda2/bin/python
# encoding: utf-8

'''
定时执行任务
'''
import time
import os

# 睡眠时间
time.sleep(3600 * 5)

# 执行命令
# os.system('python test_baseline_splitby_date.py')

os.system('python transform.py')

