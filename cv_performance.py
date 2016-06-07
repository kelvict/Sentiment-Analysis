# -*- coding: utf-8 -*-
import os

from performance import performance


if __name__ == '__main__':
    output_dir = 'data' + os.sep +'coae2014' + os.sep + 'train_nfolds'
    class_dict = {'1':'neg','2':'pos'}
    fold_num = 5
    result_dict = performance.demo_cv_performance(output_dir,fold_num,class_dict)

    ss = ''
    for key in ['p_neg','r_neg','f1_neg','p_pos','r_pos','f1_pos','macro_f1','acc']:
        ss += str(round(result_dict[key]*100,4))+'%\t'
    print ss.rstrip('\t')
    with open(output_dir + os.sep + 'result.txt', 'a') as xs:
        xs.write(ss.rstrip('\t') + '\n')
