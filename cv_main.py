from Mbsa import Mbsa
import os
import tools

if __name__ == '__main__':
    token_gram='uni'
    token_df = 2
    token_fs_opt = 1
    token_fs_method = 'WLLR'
    token_fs_num = 10000

    pos_gram = 'none'
    pos_df = 2
    pos_fs_opt = 0
    pos_fs_method = 'WLLR'
    pos_fs_num = 100000

    character_gram = 'none'
    character_df = 4
    character_fs_opt = 0
    character_fs_method = 'WLLR'
    character_fs_num = 100000

    term_weight = 'BOOL'
    rule_feature = 0

    token_fname_list = ['neg_fenci','pos_fenci']
    pos_fname_list = ['neg_pos','pos_pos']
    class_fname_list = ['neg', 'pos']


    senti_dict_lst = []
    # for senti_dict_dir in ['hownet','ntusd','tsing','emotA','emotB','cliwc']:
    #     print senti_dict_dir
    #     senti_dict = tools.load_lexicon('DICT'+os.sep+senti_dict_dir+os.sep+'senti_all')
    #     senti_dict_lst.append(senti_dict)

    test = Mbsa(token_gram, token_fs_opt, token_fs_method, token_fs_num, token_df, pos_gram,
            pos_fs_opt, pos_fs_method, pos_fs_num, pos_df, character_gram, character_fs_opt,
            character_fs_method, character_fs_num, character_df, term_weight, rule_feature)

    # input_dir = 'nlpcc_emotion' + os.sep + 'test'
    input_dir = 'data' + os.sep +'coae2014' + os.sep + 'train'
    fold_num = 5

    test.N_folds_samps(input_dir,fold_num,token_fname_list,pos_fname_list,class_fname_list)
    test.N_folds_validation(input_dir,fold_num)
