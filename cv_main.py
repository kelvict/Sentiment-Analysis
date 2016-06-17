import os
import sys
import tools

from Mbsa import Mbsa
from performance import performance

if __name__ == '__main__':
    # token_gram='uni'
    # token_df = 2
    # token_fs_opt = 1
    # token_fs_method = 'WLLR'
    # token_fs_num = 10000

    # pos_gram = 'none'
    # pos_df = 2
    # pos_fs_opt = 0
    # pos_fs_method = 'WLLR'
    # pos_fs_num = 100000

    # character_gram = 'none'
    # character_df = 4
    # character_fs_opt = 0
    # character_fs_method = 'WLLR'
    # character_fs_num = 100000

    token_gram = {
        'uni': {
            'df': 1,
            'fs_opt': 0,
            'fs_method': 'IG',
            'fs_num': 1000,
        },
        # 'bis':{
        #     'df': 1,
        #     'fs_opt': 1,
        #     'fs_method': 'IG',
        #     'fs_num': 2000,
        # },

        # 'tri':{
        #     'df': 1,
        #     'fs_opt': 1,
        #     'fs_method': 'WLLR',
        #     'fs_num': 1000,
        # }

    }

    pos_gram = {
        # 'uni': {
        #     'df': 1,
        #     'fs_opt': 1,
        #     'fs_method': 'IG',
        #     'fs_num': 20,
        # },
        # 'bis':{
        #     'df': 1,
        #     'fs_opt': 1,
        #     'fs_method': 'WLLR',
        #     'fs_num': 50,
        # }
    }

    tag_gram = {
        # 'uni': {
        #     'df': 2,
        #     'fs_opt': 1,
        #     'fs_method': 'WLLR',
        #     'fs_num': 1000,
        # },
        # 'bis':{
        #     'df': 2,
        #     'fs_opt': 1,
        #     'fs_method': 'WLLR',
        #     'fs_num': 1000,
        # }
    }

    character_gram = {
        # 'tri': {
        #     'df': 2,
        #     'fs_opt': 1,
        #     'fs_method': 'WLLR',
        #     'fs_num': 1000,
        # },
        # 'quat': {
        #     'df': 2,
        #     'fs_opt': 1,
        #     'fs_method': 'WLLR',
        #     'fs_num': 1000,
        # },
        # 'five': {
        #     'df': 2,
        #     'fs_opt': 1,
        #     'fs_method': 'WLLR',
        #     'fs_num': 2000,
        # },
        # 'six': {
        #     'df': 2,
        #     'fs_opt': 1,
        #     'fs_method': 'WLLR',
        #     'fs_num': 2000,
        # },
    }

    term_weight = 'BOOL'
    rule_feature = 1

    # token_fname_list = ['neg_fenci','pos_fenci']
    # pos_fname_list = ['neg_pos','pos_pos']
    # class_fname_list = ['neg', 'pos']

    # token_fname_list = ['neg_raw_fenci','pos_raw_fenci']
    # pos_fname_list = ['neg_raw_pos','pos_raw_pos']
    # tag_fname_list = ['neg_raw_cobine','pos_raw_cobine']
    # class_fname_list = ['neg', 'pos']

    token_fname_list = ['against_raw_fenci','favor_raw_fenci','none_raw_fenci']
    pos_fname_list = ['against_raw_pos','favor_raw_pos','none_raw_pos']
    tag_fname_list = ['against_raw_cobine','favor_raw_cobine','none_raw_cobine']
    class_fname_list = ['against', 'favor' ,'none']


    classifier_list = ['nb']
    # classifier_list = ['lg', 'nb', 'svm']
    # classifier_list = ['lg','svm']

    senti_dict_lst = []
    # for senti_dict_dir in ['hownet', 'ntusd', 'tsing', 'emotA', 'emotB', 'cliwc']:
    #     print senti_dict_dir
    #     senti_dict = tools.load_lexicon('DICT'+os.sep+senti_dict_dir+os.sep+'senti_all', float)
    #     senti_dict_lst.append(senti_dict)

    # test = Mbsa(token_gram, token_fs_opt, token_fs_method, token_fs_num, token_df, pos_gram,
    #         pos_fs_opt, pos_fs_method, pos_fs_num, pos_df, character_gram, character_fs_opt,
    #         character_fs_method, character_fs_num, character_df, term_weight, rule_feature)
    for j in range(1, 2):
        # token_gram['uni']['fs_num'] = j * 1000
        # token_gram['bis']['fs_num'] = j * 500
        # token_gram['tri']['fs_num'] = j * 1000
        # pos_gram['uni']['fs_num'] = j * 5
        # pos_gram['bis']['fs_num'] = j * 5
        test = Mbsa(token_gram, pos_gram, tag_gram, character_gram, term_weight, rule_feature, senti_dict_lst)
        # print type(test.senti_dict_lst)
        # input_dir = 'data' + os.sep +'nlpcc_emotion' + os.sep + 'test'
        # input_dir = 'data' + os.sep +'coae2014' + os.sep + 'train'
        # input_dir = 'data' + os.sep +'coae2015' + os.sep + 'train'
        # input_dir = 'data' + os.sep +'nlpcc_sentence'

        fold_num = 5

        data_dir = 'data_boson'
        data_dir = 'nlpcc_2016'
        # input_dir = str(sys.argv[1])
        # print "data_dir=", input_dir
        # for item in ['coae2015', 'nlpcc_sentence']:
        # for item in ['cobine_2']:
        # for item in ['coae2015']:
        for item in ['t1']:
            input_dir = data_dir + os.sep + item
            output_dir = data_dir + os.sep + item + '_nfolds'

            # input_dir = 'data_nlpir' + os.sep + item
            # output_dir = 'data_nlpir' + os.sep + item + '_nfolds'

            test.N_folds_samps(input_dir, fold_num,token_fname_list,pos_fname_list, tag_fname_list, class_fname_list)
            test.N_folds_validation(input_dir, fold_num, classifier_list)

            class_dict = {'1':'neg','2':'pos'}

            # classifier_list_plus = ['lg', 'nb', 'svm', 'mixed']
            classifier_list_plus = classifier_list
            for c in classifier_list_plus:
                result_dict = performance.demo_cv_performance(output_dir, fold_num, class_dict, c)
                ss = ''
                for key in ['p_neg','r_neg','f1_neg','p_pos','r_pos','f1_pos','macro_f1','acc']:
                    ss += str(round(result_dict[key]*100,4))+'%\t'
                print ss.rstrip('\t')
                with open(data_dir + os.sep + 'lord_result.txt', 'a') as xs:
                    # xs.write(item + " " + c + " uni_token_num = " + str(token_gram['uni']['fs_num']) + '\n')
                    # xs.write(item + " " + c + " bis_token_num = " + str(token_gram['bis']['fs_num']) + '\n')
                    # xs.write(item + " " + c + " uni_pos_num = " + str(pos_gram['uni']['fs_num']) + '\n')
                    # xs.write(item + " " + c + " bis_pos_num = " + str(pos_gram['bis']['fs_num']) + '\n')
                    # xs.write(item + " uni_token_num = all" + '\n')
                    # xs.write(item + " bis_token_num = " + str(token_gram['bis']['fs_num']) + '\n')
                    # xs.write(item + " tri_token_num = " + str(token_gram['tri']['fs_num']) + '\n')
                    # xs.write(item + " uni_pos_num = " + str(pos_gram['uni']['fs_num']) + '\n')
                    # xs.write("\nwith mi_senti_dict features\n")
                    xs.write(item + ' classifier ' + c + ' with all rule feature' + '\n')
                    # xs.write(item + ' without rule feature' + '\n')
                    # xs.write(item + ' all rule feature - (compare things,...)' + '\n')
                    xs.write(ss.rstrip('\t') + '\n')


