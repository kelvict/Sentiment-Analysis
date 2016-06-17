# -*- coding: utf-8 -*-
"""
@date: 2016.5.9
@author: JieJ

"""
import os

import pytc
import tools
from performance import performance


class Mbsa(object):
    def __init__(self, token_gram = 'uni', token_fs_opt = 0, token_fs_method = 'WLLR', token_fs_num = -1, token_df = 2,
        pos_gram='none', pos_fs_opt = 0, pos_fs_method = 'WLLR', pos_fs_num = -1, pos_df = 2,
        character_gram = 'none', character_fs_opt = 1, character_fs_method = 'WLLR', character_fs_num = 3000, character_df = 2,
        term_weight='BOOL', rule_feature=0, senti_dict_lst=[]):
        self.token_gram = token_gram
        self.token_fs_opt = token_fs_opt
        self.token_fs_method = token_fs_method
        self.token_fs_num = token_fs_num
        self.token_df = token_df

        self.pos_gram = pos_gram
        self.pos_fs_opt = pos_fs_opt
        self.pos_fs_method = pos_fs_method
        self.pos_fs_num = pos_fs_num
        self.pos_df = pos_df

        self.character_gram = character_gram
        self.character_fs_opt = character_fs_opt
        self.character_fs_method = character_fs_method
        self.character_fs_num = character_fs_num
        self.character_df = character_df

        self.term_weight = term_weight
        self.rule_feature = rule_feature
        self.senti_dict_lst = senti_dict_lst

    def gen_doc_terms_list(self, input_dir, token_fname_list, pos_fname_list, class_fname_list, train_opt = 0):
        doc_class_list = []

        # token ngram
        doc_token_list, token_set = [], []
        if self.token_gram != 'none':
            doc_str_token, doc_class_list = pytc.read_annotated_data([input_dir + \
                os.sep + x for x in token_fname_list], class_fname_list)
            doc_token_list = pytc.gen_N_gram(doc_str_token,self.token_gram)
            token_set = pytc.get_term_set(doc_token_list)

        # pos ngram
        doc_pos_list, pos_set = [], []
        if self.pos_gram != 'none':
            doc_str_pos, doc_class_list = pytc.read_annotated_data([input_dir + os.sep + x \
                for x in pos_fname_list], class_fname_list)
            doc_pos_list = pytc.gen_N_gram(doc_str_pos,self.pos_gram)
            pos_set = pytc.get_term_set(doc_pos_list)

        # character ngram
        doc_character_list, character_set = [], []
        if self.character_gram != 'none':
            doc_character_list = pytc.gen_character_ngram_list(doc_str_token)
            character_set = pytc.get_term_set(doc_character_list)

        # get joint doc instances set
        doc_terms_list = pytc.get_joint_sets(doc_token_list, doc_pos_list)
        doc_terms_list = pytc.get_joint_sets(doc_terms_list, doc_character_list)

        # if it's in training producure
        if train_opt == 1:

            if len(doc_token_list) > 0:
                print "len(token_set)=",len(token_set)
                token_set = pytc.feature_selection_all(doc_token_list, doc_class_list, class_fname_list,
                    token_set, self.token_fs_opt, self.token_df, self.token_fs_method, self.token_fs_num)
                print "after feature selection, len(token_set) =", len(token_set)

            if len(doc_pos_list) > 0:
                print "len(pos_set)=",len(pos_set)
                pos_set = pytc.feature_selection_all(doc_pos_list, doc_class_list, class_fname_list,
                    pos_set, self.pos_fs_opt, self.pos_df, self.pos_fs_method, self.pos_fs_num)
                print "after feature selection, len(pos_set) =", len(pos_set)

            if len(doc_character_list) > 0:
                print "len(character_set)=", len(character_set)
                character_set = pytc.feature_selection_all(doc_character_list, doc_class_list,
                    class_fname_list, character_set, self.character_fs_opt, self.character_df,
                    self.character_fs_method, self.character_fs_num)
                print "len(character_set) =", len(character_set)

        # get joint term set
        term_set = token_set + pos_set
        term_set = term_set + character_set

        return doc_class_list, doc_str_token, doc_terms_list, term_set

    def gen_train_samps(self, train_dir, train_samp_dir, term_set_dir, token_fname_list,
            pos_fname_list, class_fname_list):

        fname_samps_train = train_samp_dir +os.sep+'train.samp'
        fname_term_set_train = term_set_dir +os.sep+'term.set'

        doc_class_list_train, doc_str_token_train, doc_terms_list_train, term_set_train = \
        self.gen_doc_terms_list(train_dir, token_fname_list, pos_fname_list, class_fname_list, train_opt = 1)

        pytc.save_term_set(term_set_train, fname_term_set_train)

        # unigram的token，单独作为参数用来构建其他规则特征
        doc_uni_token_train = pytc.gen_N_gram(doc_str_token_train,'uni')

        term_dict = dict(zip(term_set_train, range(1,len(term_set_train)+1)))
        class_dict = dict(zip(class_fname_list, range(1,1+len(class_fname_list))))

        if self.term_weight=='TFIDF':
            doc_num_train = len(doc_class_list_train)
            df_term_train = pytc.stat_df_term(term_set_train,doc_terms_list_train)
            idf_term_train = pytc.stat_idf_term(doc_num_train,df_term_train)
        else:
            idf_term_train = []

        print "building samps......"
        samp_list_train, class_list_train = pytc.build_samps(term_dict, class_dict, doc_class_list_train,
        doc_terms_list_train, doc_uni_token_train, self.term_weight, self.rule_feature, self.senti_dict_lst)

        print "saving samps......"
        pytc.save_samps(samp_list_train, class_list_train, fname_samps_train)


    def gen_test_samps(self, test_dir, term_set_dir, test_samp_dir, result_file_dir, token_fname_list,
        pos_fname_list, class_fname_list):

        fname_term_set = term_set_dir + os.sep + 'term.set'
        fname_samps_test = test_samp_dir +os.sep+'test.samp'

        if not os.path.isfile(fname_term_set):
            print "cant find term set file."
            return

        if not os.path.exists(test_dir):
            print "test dir does not exist."
            return

        doc_class_list_test, doc_str_token_test, doc_terms_list_test, term_set_test = \
        self.gen_doc_terms_list(test_dir, token_fname_list, pos_fname_list, class_fname_list, train_opt = 0)

        # unigram的token，用来构建其他规则特征
        doc_uni_token_test = pytc.gen_N_gram(doc_str_token_test,'uni')

        term_set_train = pytc.load_term_set(fname_term_set)
        term_dict = dict(zip(term_set_train, range(1,len(term_set_train)+1)))
        class_dict = dict(zip(class_fname_list, range(1,1+len(class_fname_list))))
        class_dict['test'] = 0

        if self.term_weight=='TFIDF':
            doc_num_test = len(doc_class_list_test)
            df_term_test = pytc.stat_df_term(term_set_test,doc_terms_list_test)
            idf_term_test = pytc.stat_idf_term(doc_num_test,df_term_test)
        else:
            idf_term_test = []

        print "building samps......"
        samp_list_test, class_list_test = pytc.build_samps(term_dict, class_dict, doc_class_list_test,
        doc_terms_list_test, doc_uni_token_test, self.term_weight, self.rule_feature, self.senti_dict_lst)

        print "saving samps......"
        pytc.save_samps(samp_list_test, class_list_test, fname_samps_test)

    def N_folds_samps(self,input_dir,fold_num, token_fname_list, pos_fname_list,class_fname_list):
        '''将语料按照交叉验证的折数进行分割'''
        output_dir = input_dir+'_nfolds'
        class_dict = dict(zip(token_fname_list, [str(i) for i in range(1, len(token_fname_list) + 1)]))
        #将原文件、分词文件、词性文件按照n折交叉验证分割
        pytc.gen_nfolds_f2(input_dir, output_dir, fold_num, token_fname_list, class_dict)
        if self.pos_gram != 'none':
            pytc.gen_nfolds_f2(input_dir, output_dir, fold_num, pos_fname_list, class_dict)

        for fold_id in range(1, fold_num+1):
            print '\n\n##### Cross Validation: fold' + str(fold_id) + ' #####'
            fold_dir = output_dir + os.sep + 'fold' + str(fold_id)
            fold_train_dir = fold_dir + os.sep + 'train'
            fold_test_dir = fold_dir + os.sep + 'test'
            train_samp_dir, term_set_dir = fold_train_dir, fold_train_dir
            test_samp_dir, result_file_dir = fold_test_dir, fold_test_dir

            test_token_fname = ['test_fenci']
            test_pos_fname = ['test_pos']
            test_class_fname = ['test']
            self.gen_train_samps(fold_train_dir,train_samp_dir, term_set_dir, token_fname_list, pos_fname_list, class_fname_list)
            self.gen_test_samps(fold_test_dir,term_set_dir, test_samp_dir, result_file_dir, test_token_fname, test_pos_fname, test_class_fname)


    def N_folds_validation(self,input_dir,fold_num):
        '''对每折验证中的语料进行训练与测试，并求融合模型的平均正确率'''
        output_dir = input_dir+'_nfolds'
        # mix_acc_list = []
        for fold_id in range(1,fold_num+1):
            fold_dir = output_dir + os.sep + 'fold' + str(fold_id)
            train_samp_dir, model_file_dir = fold_dir + os.sep + 'train', fold_dir + os.sep + 'train'
            test_samp_dir, result_file_dir = fold_dir + os.sep + 'test', fold_dir + os.sep + 'test'

            pytc.liblinear_learn(train_samp_dir, model_file_dir, learn_opt='-s 7 -c 1')
            pytc.liblinear_predict(test_samp_dir, model_file_dir, result_file_dir, classify_opt='-b 1')

            # mix_acc = self.mix_model(ml_file_dir,mix_classifier_list)
            # mix_acc_list.append(mix_acc)
        # print "\nThe avg of mix_acc is",sum(mix_acc_list)*1.0/len(mix_acc_list)

if __name__ == '__main__':
    token_gram = {
        'uni': {
            'df': 2,
            'fs_opt': 1,
            'fs_method': 'WLLR',
            'fs_num': 100000,
        },
        'bis':{
            'df': 2,
            'fs_opt': 1,
            'fs_method': 'WLLR',
            'fs_num': 100000,
        }

    }

    # token_gram = 'uni'
    # token_df = 2
    # token_fs_opt = 1
    # token_fs_method = 'WLLR'
    # token_fs_num = 100000

    # pos_gram = 'none'
    # # pos_gram = 'uni'
    # pos_df = 2
    # pos_fs_opt = 1
    # pos_fs_method = 'WLLR'
    # pos_fs_num = 1000

    # character_gram = 'none'
    # character_df = 4
    # character_fs_opt = 0
    # character_fs_method = 'WLLR'
    # character_fs_num = 100000

    term_weight = 'BOOL'
    rule_feature = 1


    senti_dict_lst = []
    # for senti_dict_dir in ['hownet','ntusd','tsing','emotA','emotB','cliwc', 'distant']:
    #     print senti_dict_dir
    #     senti_dict = tools.load_lexicon('DICT'+os.sep+senti_dict_dir+os.sep+'senti_all', float)
    #     senti_dict_lst.append(senti_dict)

    # test = Mbsa(token_gram, token_fs_opt, token_fs_method, token_fs_num, token_df, pos_gram,
    #         pos_fs_opt, pos_fs_method, pos_fs_num, pos_df, character_gram, character_fs_opt,
    #         character_fs_method, character_fs_num, character_df, term_weight, rule_feature, senti_dict_lst)

    test = Mbsa(token_gram, term_weight, rule_feature, senti_dict_lst)

    ''' train & test '''
    train_dir = 'data' + os.sep + 'nlpcc_emotion' + os.sep + 'train'
    # train_dir = 'data' + os.sep + 'coae2014' + os.sep + 'train_nfolds' + os.sep +'fold2' + os.sep + 'train'
    train_samp_dir = train_dir
    term_set_dir = train_dir
    model_file_dir = train_dir

    token_fname_list = ['neg_fenci','pos_fenci']
    pos_fname_list = ['neg_pos','pos_pos']
    class_fname_list = ['neg', 'pos']

    test.gen_train_samps(train_dir, train_samp_dir, term_set_dir,
        token_fname_list, pos_fname_list, class_fname_list)
    pytc.liblinear_learn(train_samp_dir, model_file_dir, learn_opt='-s 7 -c 1')

    test_dir = 'data' + os.sep + 'nlpcc_emotion' + os.sep + 'test'
    # test_dir = 'data' + os.sep + 'coae2014' + os.sep + 'train_nfolds' + os.sep +'fold2' + os.sep + 'test'
    test_samp_dir = test_dir
    result_file_dir = test_dir

    token_fname_list = ['test_fenci']
    pos_fname_list = ['test_pos']
    class_fname_list = ['test']

    test.gen_test_samps(test_dir, term_set_dir, test_samp_dir, result_file_dir, token_fname_list, pos_fname_list, class_fname_list)
    pytc.liblinear_predict(test_samp_dir, model_file_dir, result_file_dir, classify_opt='-b 1')


    '''performance'''
    label = [x.strip() for x in open(test_dir+os.sep+'test_label').readlines()]
    result = [x.strip().split()[0] for x in open(result_file_dir+os.sep+'lg.result').readlines()[1:]]

    class_dict = {'1':'neg','2':'pos'}
    result_dict = performance.demo_performance(result,label,class_dict)

    print len(label)
    print str(label.count('1'))+'\t'+str(label.count('2'))

    ss = ''
    for key in ['p_neg','r_neg','f1_neg','p_pos','r_pos','f1_pos','macro_f1','acc']:
        ss += str(round(result_dict[key]*100,4))+'%\t'
    print ss.rstrip('\t')
    print "#"

