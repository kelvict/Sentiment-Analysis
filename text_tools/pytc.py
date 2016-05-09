# -*- coding: utf-8 -*-

from __future__ import division
import os
import re
import random
import math

########### Global Parameters ###########

TOOL_PATH = '.'
LIBLINEAR_LEARN_EXE = TOOL_PATH + os.sep+ 'liblinear-1.96' + os.sep + 'windows' + os.sep + 'train.exe'
LIBLINEAR_CLASSIFY_EXE = TOOL_PATH + os.sep+ 'liblinear-1.96' + os.sep + 'windows' + os.sep + 'predict.exe'

LOG_LIM = 1E-300

########## Data I/O Functions ##########

def read_annotated_data(fname_list, class_list):
    '''
    read data with class annotation, one class per file, one instance per line
    return instance list and corresponding class label list
    '''
    doc_str_list = []
    doc_class_list = []
    for doc_fname,class_fname in zip(fname_list, class_list):
        doc_str_list_one_class = [x.strip() for x in open(doc_fname, 'r').readlines()]
        doc_str_list.extend(doc_str_list_one_class)
        doc_class_list.extend([class_fname] * len(doc_str_list_one_class))
    return doc_str_list,doc_class_list

def read_unannotated_data(fname_list):
    '''
    read data without class annotation, one instance per line
    return instance list
    '''
    doc_str_list = []
    for doc_fname in fname_list:
        doc_str_list_one_class = [x.strip() for x in open(doc_fname, 'r').readlines()]
        doc_str_list.extend(doc_str_list_one_class)
    return doc_str_list


########## Feature Extraction & Storage Fuctions ##########

def get_doc_unis_list(doc_str_list):
    '''generate unigram language model for each segmented instance'''
    unis_list = [x.split() for x in doc_str_list]
    return unis_list

def get_doc_bis_list(doc_str_list):
    '''generate bigram language model for each segmented instance'''
    unis_list = get_doc_unis_list(doc_str_list)
    doc_bis_list = []
    for k in range(len(doc_str_list)):
        unis = unis_list[k]
        if len(unis) == 0:
            doc_bis_list.append([])
            continue
        unis_pre, unis_after = ['<bos>'] + unis, unis + ['<eos>']
        doc_bis_list.append([x + '<w-w>' + y for x, y in zip(unis_pre, unis_after)])
    return doc_bis_list

def get_doc_triple_list(doc_str_list):
    '''generate triple-gram language model for each segmented instance'''
    doc_unis_list = get_doc_unis_list(doc_str_list)
    doc_bis_list = get_doc_bis_list(doc_str_list)
    doc_triple_list = []
    for k in range(len(doc_str_list)):
        unis = doc_unis_list[k]
        bis = doc_bis_list[k]
        if len(bis)<=2:
            doc_triple_list.append([])
            continue
        pre, after = bis[:-1], unis[1:] + ['<eos>']
        doc_triple_list.append([x + '<w-w>' + y for x, y in zip(pre, after)])
    return doc_triple_list

def get_doc_quat_list(doc_str_list):
    '''generate triple-gram language model for each segmented instance'''
    doc_unis_list = get_doc_unis_list(doc_str_list)
    doc_bis_list = get_doc_bis_list(doc_str_list)
    doc_triple_list = get_doc_triple_list(doc_str_list)
    doc_quat_list = []
    for k in range(len(doc_str_list)):
        unis = doc_unis_list[k]
        bis = doc_bis_list[k]
        triple = doc_triple_list[k]
        if len(triple)<=2:
            doc_quat_list.append([])
            continue
        pre, after = ['<bos>'] + unis[:-2], triple[1:]
        doc_quat_list.append([x+'<w-w>'+y for x,y in zip(pre,after)])
    return doc_quat_list

def get_joint_sets(lst1, lst2):
    '''
    map corresponding element for two 2-dimention list
    '''
    if len(lst1) != len(lst2):
        print "different lengths, return the first list object"
        return lst1
    return map(lambda x, y : x + y, lst1, lst2)

def gen_N_gram(doc_str_list, ngram='uni'):
    '''
    generating NGRAM for each instance according to given N
    '''
    doc_ngram_list = []
    if ngram=='uni':
        doc_ngram_list = get_doc_unis_list(doc_str_list)
    elif ngram=='bis':
        doc_uni_list = get_doc_unis_list(doc_str_list)
        doc_bis_list = get_doc_bis_list(doc_str_list)
        doc_ngram_list = get_joint_sets(doc_uni_list, doc_bis_list)
    elif ngram=='tri':
        doc_uni_list = get_doc_unis_list(doc_str_list)
        doc_bis_list = get_doc_bis_list(doc_str_list)
        doc_trip_list = get_doc_triple_list(doc_str_list)
        tmp = get_joint_sets(doc_uni_list, doc_bis_list)
        doc_ngram_list = get_joint_sets(tmp,doc_trip_list)
    elif ngram=='quat':
        doc_uni_list = get_doc_unis_list(doc_str_list)
        doc_bis_list = get_doc_bis_list(doc_str_list)
        doc_trip_list = get_doc_triple_list(doc_str_list)
        doc_quat_list = get_doc_quat_list(doc_str_list)
        tmp1 = get_joint_sets(doc_uni_list, doc_bis_list)
        tmp2 = get_joint_sets(tmp1, doc_trip_list)
        doc_ngram_list = get_joint_sets(tmp2,doc_quat_list)
    else:
        for i in range(len(doc_str_list)):
            doc_ngram_list.append([])
    return doc_ngram_list

def get_term_set(doc_terms_list):
    '''generate unique term set fron N segmented instances, N = len(doc_terms_list) '''
    term_set = set()
    for doc_terms in doc_terms_list:
        term_set.update(doc_terms)
    return sorted(list(term_set))

def save_term_set(term_set, fname):
    '''save term set'''
    open(fname, 'w').writelines([x + '\n' for x in term_set])

def load_term_set(fname):
    '''load term set'''
    term_set = [x.strip() for x in open(fname, 'r').readlines()]
    return term_set



########## Building Sample Files ##########

# def build_samps(term_dict, class_dict, doc_terms_list,doc_uni_token,doc_uni_pos, doc_class_list,
#                 senti_dict_lst,term_weight, rule_feature=0, idf_term=None):
def build_samps(term_dict, class_dict, doc_class_list, doc_terms_list, term_weight, rule_feature=0):
    samp_dict_list = []
    samp_class_list = []

    # # 初始化情感分析对象, 该对象使用带有强度标注的情感词典进行分析
    # win_size = 4
    # phrase_size = 3
    # test = Lbsa(win_size,phrase_size)

    for k in range(len(doc_class_list)):
        doc_class = doc_class_list[k]
        samp_class = class_dict[doc_class]
        samp_class_list.append(samp_class)
        doc_terms = doc_terms_list[k]
        samp_dict = {}
        for term in doc_terms:
            if term_dict.has_key(term):
                term_id = term_dict[term]
                if term_weight == 'BOOL':
                    samp_dict[term_id] = 1
                elif term_weight == 'TF':
                    if samp_dict.has_key(term_id):
                        samp_dict[term_id] += 1
                    else:
                        samp_dict[term_id] = 1
                elif term_weight == 'TFIDF':
                    if samp_dict.has_key(term_id):
                        samp_dict[term_id] += idf_term[term]
                    else:
                        samp_dict[term_id] = idf_term[term]

        # if rule_feature!=0:
        #     doc_tokens = doc_uni_token[k]
        #     fixed_id = len(term_dict)+1        #下一个特征开始的ID号

        #     rule_result = test.cal_document(doc_tokens,'none')
        #     #########               添加规则情感特征                 ###########
        #     # 先添加情感词以外的特征，包括：
        #     # 否定词数量，程度副词数量，感叹词数量，第一人称词数量，第二人称词数量
        #     for key in ['deny_ct','degree_ct']:
        #         val = rule_result[key]
        #         if float(val) !=0:
        #             samp_dict[fixed_id] = float(val)
        #         fixed_id += 1

        #     yuqici_ct = doc_tokens.count('啊')+doc_tokens.count('啦')+doc_tokens.count('呀')+\
        #     doc_tokens.count('吧')+doc_tokens.count('哇')+doc_tokens.count('哦')
        #     if yuqici_ct>0:
        #         samp_dict[fixed_id] = float(yuqici_ct)
        #     fixed_id += 1

        #     if '我' in doc_terms or '我们' in doc_terms:
        #         samp_dict[fixed_id] = 1
        #     fixed_id += 1
        #     if '你' in doc_terms or '你们' in doc_terms:
        #         samp_dict[fixed_id] = 1
        #     fixed_id += 1

        #     # 标点符号特征
        #     for punc in ['!','?','！','！！！','？','？？？','。。。']:
        #         punc_ct = doc_tokens.count(punc)
        #         if punc_ct>0:
        #             samp_dict[fixed_id] = float(punc_ct)
        #         fixed_id += 1

        #     # 添加情感词相关特征
        #     for key in ['pos_ct','neg_ct','pos_sub','neg_sub','sub_ct', 'score','face_score','final_score']:
        #         val = rule_result[key]
        #         if float(val) !=0:
        #             samp_dict[fixed_id] = float(val)
        #         fixed_id += 1

        #     #正向情感词数量是否多于消极情感词
        #     fixed_id = trans_feature_weight(rule_result['pos_ct'],rule_result['neg_ct'],fixed_id,samp_dict)
        #     #正向子句数是否多于负向子句数
        #     fixed_id = trans_feature_weight(rule_result['pos_sub'],rule_result['neg_sub'],fixed_id,samp_dict)

        #     score = switch_final_score(rule_result['final_score'])
        #     face_score = switch_final_score(rule_result['face_score'])
        #     fixed_id = trans_feature_weight(score,2,fixed_id,samp_dict)
        #     fixed_id = trans_feature_weight(face_score,2,fixed_id,samp_dict)

        #     for senti_dict in senti_dict_lst:
        #         # pos_word_num, neg_word_num = test.general_lex_method(doc_tokens, senti_dict, 4)
        #         pos_word_num, neg_word_num = test.character_ngram_method(doc_tokens, senti_dict, 4)
        #         if pos_word_num>0:
        #             samp_dict[fixed_id] = pos_word_num
        #         fixed_id += 1
        #         if neg_word_num>0:
        #             samp_dict[fixed_id] = neg_word_num
        #         fixed_id += 1
        #         fixed_id = trans_feature_weight(pos_word_num, neg_word_num,fixed_id,samp_dict)
        ###########################################################################
        samp_dict_list.append(samp_dict)
    return samp_dict_list, samp_class_list

def save_samps(samp_dict_list, samp_class_list, fname, feat_num=0):
    length = len(samp_class_list)
    fout = open(fname, 'w')
    for k in range(length):
        samp_dict = samp_dict_list[k]
        samp_class = samp_class_list[k]
        fout.write(str(samp_class) + '\t')
        for term_id in sorted(samp_dict.keys()):
            if feat_num == 0 or term_id < feat_num:
                fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
        fout.write('\n')
    fout.close()

########## Learning & Classification Functions Using Machine Learning Tools##########

def liblinear_learn(train_samp_dir, model_file_dir, learn_opt='-s 7 -c 1'):


    fname_samp_train = train_samp_dir + os.sep + 'train.samp'
    fname_model = model_file_dir + os.sep + 'lg.model'

    import subprocess
    pop = subprocess.Popen(LIBLINEAR_LEARN_EXE + ' ' +  learn_opt + ' ' + \
    fname_samp_train + ' ' + fname_model)
    pop.wait()

def liblinear_predict(test_samp_dir, model_file_dir, result_file_dir, classify_opt='-b 1'):
    fname_model =model_file_dir + os.sep + 'lg.model'
    fname_samp_test = test_samp_dir + os.sep + 'test.samp'
    fname_result = result_file_dir + os.sep + 'lg.result'

    import subprocess
    pop = subprocess.Popen(LIBLINEAR_CLASSIFY_EXE + ' ' + classify_opt + ' ' \
            + fname_samp_test + ' ' + fname_model + ' ' + fname_result)
    pop.wait()
