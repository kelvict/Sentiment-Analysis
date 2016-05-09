# -*- coding: utf-8 -*-
"""
@date: 2016.5.9
@author: JieJ

"""
import os

from text_tools import pytc
from performance import performance


class Mbsa(object):
    def __init__(self, token_gram = 'uni', token_fs_opt = 0, token_fs_method = 'WLLR', token_fs_num = -1, token_df = 2,
        pos_gram='none', pos_fs_opt = 0, pos_fs_method = 'WLLR', pos_fs_num = -1, pos_df = 2,
        character_gram = 'none', character_fs_opt = 1, character_fs_method = 'WLLR', character_fs_num = 3000, character_df = 2,
        term_weight='BOOL', rule_feature=0):
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

    def gen_doc_terms_list(self, input_dir, token_fname_list, pos_fname_list, class_fname_list, train_opt = 0):
        doc_class_list = []

        # token ngram
        doc_token_list = []
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
            print "len(token_set)=",len(token_set)
            token_set = pytc.feature_selection_all(doc_token_list, doc_class_list, class_fname_list,
                token_set, self.token_fs_opt, self.token_df, self.token_fs_method, self.token_fs_num)
            print "after feature selection, len(token_set)=", len(token_set)

            print "len(pos_set)=",len(pos_set)
            pos_set = pytc.feature_selection_all(doc_pos_list, doc_class_list, class_fname_list,
                pos_set, self.pos_fs_opt, self.pos_df, self.pos_fs_method, self.pos_fs_num)
            print "after feature selection, len(pos_set)=", len(pos_set)

            print "len(character_set)=", len(character_set)
            character_set = pytc.feature_selection_all(doc_character_list, doc_class_list,
                class_fname_list, character_set, self.character_fs_opt, self.character_df,
                self.character_fs_method, self.character_fs_num)
            print "len(character_set)=", len(character_set)

        # get joint term set
        term_set = token_set + pos_set
        term_set = term_set + character_set

        return doc_class_list, doc_str_token, doc_terms_list, term_set

    def gen_train_samps(self, train_dir, train_samp_dir, term_set_dir, token_fname_list,
            pos_fname_list, class_fname_list):

        fname_samps_train = train_samp_dir +os.sep+'train.samp'
        fname_term_set_train = term_set_dir +os.sep+'term.set'

        doc_class_list_train, doc_str_token_train, doc_terms_list_train, term_set_train = \
        self.gen_doc_terms_list(train_dir, token_fname_list, pos_fname_list, class_fname_list, train_opt = 0)

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
        doc_terms_list_train, self.term_weight, self.rule_feature)

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
        doc_terms_list_test, self.term_weight, self.rule_feature)

        print "saving samps......"
        pytc.save_samps(samp_list_test, class_list_test, fname_samps_test)

if __name__ == '__main__':
    token_gram='uni'
    token_df = 2
    token_fs_opt = 1
    token_fs_method = 'WLLR'
    token_fs_num = 100000

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
    rule_feature = 1


    test = Mbsa(token_gram, token_fs_opt, token_fs_method, token_fs_num, token_df, pos_gram,
            pos_fs_opt, pos_fs_method, pos_fs_num, pos_df, character_gram, character_fs_opt,
            character_fs_method, character_fs_num, character_df, term_weight, rule_feature)


    ''' train & test '''
    train_dir = 'data' + os.sep + 'nlpcc_emotion' + os.sep + 'train'
    train_samp_dir = train_dir
    term_set_dir = train_dir
    model_file_dir = train_dir

    token_fname_list = ['neg_fenci','pos_fenci']
    pos_fname_list = ['neg_pos','pos_pos']
    class_fname_list = ['neg', 'pos']


    test_dir = 'data' + os.sep + 'nlpcc_emotion' + os.sep + 'test'
    test_samp_dir = test_dir
    result_file_dir = test_dir

    token_fname_list = ['test_fenci']
    pos_fname_list = ['test_pos']
    class_fname_list = ['test']

    test.gen_train_samps(train_dir, train_samp_dir, term_set_dir,
        token_fname_list, pos_fname_list, class_fname_list)
    pytc.liblinear_learn(train_samp_dir, model_file_dir, learn_opt='-s 7 -c 1')

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


