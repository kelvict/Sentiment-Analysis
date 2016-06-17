# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 21:19:43 2015

@author: JieJ
"""

import os
import tools

from Lbsa import Lbsa
from performance import performance


if __name__ == '__main__':

    win_size = 4
    phrase_size = 3
    test = Lbsa(win_size,phrase_size)

    data_dir = 'data_boson'
    # data_dir = 'data_nlpir'

    for item in ['coae2015', 'nlpcc_sentence']:
    # for item in ['cobine_2']:
        test_dir = data_dir + os.sep + item
        output_dir = test_dir + os.sep + 'rule_sas'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        fenci_fname = 'test_raw_fenci'
        rule_score_fname = 'rule_score.txt'
        rule_result_fname = 'rule_result.txt'

        documents = [x.strip().split() for x in open(test_dir + os.sep + fenci_fname)]

        result_dict_lst = []

        print u"一共" + str(len(documents)) + u"篇文档......"
        for j in range(len(documents)):
            # result = test.cal_document(documents[j], 'none')
            result = test.simple_rule_score(documents[j])
            result_dict_lst.append(result)


        final_score = [res['final_score'] for res in result_dict_lst]

        tools.write_score_file(final_score, output_dir + os.sep + rule_score_fname)
        tools.classify_2_way(output_dir + os.sep + rule_score_fname, output_dir + os.sep + rule_result_fname, 0)


        '''performance'''
        result = [x.strip() for x in open(output_dir + os.sep + 'rule_result.txt').readlines()]
        label = [x.strip() for x in open(test_dir + os.sep + 'test_label').readlines()]
        print len(result),len(label)

        class_dict = {'1':'neg','2':'pos'}
        result_dict = performance.demo_performance(result,label,class_dict)
        ss = ''
        for key in ['p_neg','r_neg','f1_neg','p_pos','r_pos','f1_pos','macro_f1','acc']:
            ss += str(round(result_dict[key]*100,4))+'%\t'
        ss = ss.rstrip('\t')
        print ss
        with open(data_dir + os.sep + 'lord_performance.txt', 'a') as xs:
            xs.write(data_dir + ' ' + item + '\n')
            xs.write(ss + '\n')

    print 'over'



