# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:02:50 2015

@author: JieJ
"""
import os


input_dir = 'data_boson'
input_dir = 'data_nlpir'

for item in ['coae2014', 'coae2015', 'nlpcc_emotion', 'nlpcc_sentence']:
    data_dir = input_dir + os.sep + item
    neg_fenci_lines = open(data_dir + os.sep + 'neg_raw_fenci').readlines()
    pos_fenci_lines = open(data_dir + os.sep + 'pos_raw_fenci').readlines()

    test_fenci_lines = neg_fenci_lines + pos_fenci_lines
    test_label = ['1'] * len(neg_fenci_lines) + ['2'] * len(pos_fenci_lines)

    with open(data_dir + os.sep + 'test_raw_fenci', 'w') as xs, \
    open(data_dir + os.sep + 'test_label', 'w') as ws:
        xs.writelines(test_fenci_lines)
        ws.writelines([x + '\n' for x in test_label])
