# -*- coding: utf-8 -*-
"""
@author: JieJ

提供了一个基于有强度标注情感词典的规则分析方法

"""
from __future__ import division
import os

import tools


class Lbsa(object):
    '''Lexcion based sentiment analysis class'''
    dict_path = 'DICT'
    degree_dict = tools.load_lexicon(dict_path + os.sep + 'degree', float)
    senti_dict = tools.load_lexicon(dict_path + os.sep + 'senti', float)
    senti_distant_dict = tools.load_lexicon(dict_path+os.sep+'senti_distant_new.txt', float)

    face_dict = tools.load_lexicon(dict_path + os.sep + 'all_face', float)

    pre_adv_list = [x.strip() for x in open(dict_path + os.sep + 'pre_adv').readlines()]
    nounAdj_dict = tools.load_lexicon(dict_path + os.sep + 'nounAdj', float)
    pre_verb_dict = tools.load_lexicon(dict_path + os.sep + 'pre_verb', float)
    keyword_dict = tools.load_lexicon(dict_path + os.sep + 'keyword', float)

    virtual_list = [x.strip() for x in open(dict_path + os.sep + 'virtual').readlines()]
    # wish_list = [x.strip() for x in open(dict_path + os.sep + 'wishes').readlines()]
    sarcasm_list = [x.strip() for x in open(dict_path + os.sep + 'sarcasm').readlines()]
    deny_list= [x.strip() for x in open(dict_path + os.sep + 'deny').readlines()]

    puncs_list = [x.strip() for x in open(dict_path + os.sep + 'puncs').readlines()]
    sub_puncs_list = [x.strip() for x in open(dict_path + os.sep + 'sub_puncs').readlines()]

    adverse_list = [x.strip() for x in open(dict_path + os.sep + 'adverse').readlines()]
    pre_adverse_list = [x.strip() for x in open(dict_path + os.sep + 'pre_adverse').readlines()]

    def __init__(self,win_size,phrase_size):
        self.win_size = win_size
        self.phrase_size = phrase_size

    def distant_dict_score(self, doc):
        if len(doc) == 0:
            return 0, 0
        score = 0
        for term in doc:
            if term in self.senti_distant_dict:
                score += self.senti_distant_dict[term]
        avg_score = score/len(doc)
        return score, avg_score

    def your_dict_score(self, doc, your_senti_dict):
        if len(doc) == 0:
            return 0, 0, 0
        pos_ct, neg_ct, score = 0, 0, 0
        # print type(doc)
        for term in doc:
            if term in your_senti_dict:
                score += your_senti_dict[term]
                if your_senti_dict[term] > 0:
                    pos_ct += 1
                elif your_senti_dict[term] < 0:
                    neg_ct += 1
                else:
                    pass
        return pos_ct, neg_ct, score

    def simple_rule_score(self, doc):
        res_dict = {'final_score' : 0}
        if len(doc) == 0:
            return res_dict
        score = 0
        for term in doc:
            if term in self.senti_dict:
                score += self.senti_dict[term]

        face_score = self.cal_face(doc)['face_score']

        # res_dict['final_score'] = score + face_score
        res_dict['final_score'] = score
        return res_dict


    def character_ngram_method(self, doc_tokens, senti_dict, deny_win_size):
        ''' 一个基于字符ngrams的情感词检测方法 '''
        pos_ct,neg_ct = 0,0
        doc_str = ''.join(doc_tokens)
        doc_str = doc_str.replace(' ','').decode('utf8', 'ignore')
        for ngram in range(2,5):
            for i in range(len(doc_str)-ngram+1):
                term = doc_str[i:i+ngram]
                term = term.encode('utf8','ignore')
                if senti_dict.has_key(term):
                    pre_index = max(0,i-deny_win_size)   #设定窗口，向前搜索否定词
                    sub_doc_str = doc_str[pre_index:i]
                    deny_flag = self.character_ngram_deny(sub_doc_str)
                    score = senti_dict[term] * deny_flag
                    # score = senti_dict[term]
                    if score > 0:
                        pos_ct += 1
                    elif score < 0:
                        neg_ct += 1
        return pos_ct, neg_ct

    def character_ngram_deny(self, doc_str):
        ''' 一个基于字符ngrams的否定词检测方法 '''
        for ngram in [3,2]:
            for i in range(len(doc_str)-ngram+1):
                term = doc_str[i:i+ngram]
                term = term.encode('utf8','ignore')
                if term in self.deny_list:
                    # print term.decode('utf8','ignore')
                    return -1
        return 1

    def cal_subsent(self,sub_s):
        '''计算一个子分句考虑否定，加强，转折，虚拟后的情感倾向性情感极性得分'''
        win_size = self.win_size    #否定词程度词检查窗口
        score = 0                   # 总得分
        pos_ct = 0                  # 褒义情感词数量
        neg_ct = 0                  # 贬义情感词数量
        deny_ct = 0                 # 否定词数量
        degree_ct = 0               # 程度词数量
        score_detail = ''           # 具体得分项字符串
        i = 0

        while i<len(sub_s):
            if sub_s[i] in Lbsa.virtual_list:   #如果出现虚拟词，忽略该句情感
                score = 0
                score_detail = ''
                break
            elif sub_s[i] in Lbsa.senti_dict:   #如果出现情感词
                neg_flag = 0
                tmp = float(Lbsa.senti_dict[sub_s[i]]) # 记录情感词得分
                tmp_detail = sub_s[i]+'; '
                pre_index = max(0,i-win_size)          # 设定窗口，向前搜索否定词

                for j in range(i-1,pre_index,-1):
                    if sub_s[j] in Lbsa.deny_list:
                        tmp *= -1
                        deny_ct += 1
                        tmp_detail = sub_s[j] + tmp_detail
                        neg_flag = 1
                    elif sub_s[j] in Lbsa.degree_dict:
                        tmp *= float(Lbsa.degree_dict[sub_s[j]])
                        degree_ct += 1
                        tmp_detail = sub_s[j] + tmp_detail
                    else:
                        pass
                if tmp>0:
                    pos_ct += 1
                if tmp<0:
                    neg_ct += 1
                score += tmp
                score_detail += tmp_detail
                # if neg_flag == 1:
                #     sss = ' '.join(sub_s)
                #     print "#", sss.decode('utf8').encode('gbk','ignore')
                #     print tmp_detail.decode('utf8').encode('gbk','ignore')

            #如果出现情感短语前置动词
            elif sub_s[i] in Lbsa.pre_verb_dict:
                tmp_res = self.cal_phrase(sub_s, i, Lbsa.keyword_dict)
                score += tmp_res['tmp']
                i = tmp_res['end']                 # 设置i为下一个开始位置
                pos_ct += tmp_res['pos_ct']
                neg_ct += tmp_res['neg_ct']
                score_detail += tmp_res['tmp_detail']

                # if tmp_res['tmp_detail'] != '':
                #     sss = ' '.join(sub_s)
                #     print "#", sss.decode('utf8').encode('gbk','ignore')
                #     print tmp_res['tmp_detail'].decode('utf8').encode('gbk','ignore')

            #如果出现情感短语前置副词
#            elif sub_s[i] in Lbsa.pre_adv_list:
#                if i+1<len(sub_s) and sub_s[i+1] in Lbsa.nounAdj_dict:
#                    tmp = Lbsa.nounAdj_dict[sub_s[i+1]]
#                    score += tmp*1.5
#                    score_detail += sub_s[i]+sub_s[i+1]+'; '
##                    print score_detail,tmp
#                    if tmp>0:
#                        pos_ct += 1
#                    if tmp<0:
#                        neg_ct += 1

            #如果出现前置转折词
            elif sub_s[i] in Lbsa.pre_adverse_list:
                temp_res = self.cal_subsent(sub_s[i+1:])
                score += 0.5 * temp_res['score']
                pos_ct += temp_res['pos_ct']
                neg_ct += temp_res['neg_ct']
                deny_ct += temp_res['deny_ct']
                degree_ct += temp_res['degree_ct']
                score_detail += temp_res['score_detail']
                break

            #如果出现后置转折词
            elif sub_s[i] in Lbsa.adverse_list:
                temp_res = self.cal_subsent(sub_s[i+1:])
                score += 2 * temp_res['score']
                pos_ct += temp_res['pos_ct']
                neg_ct += temp_res['neg_ct']
                deny_ct += temp_res['deny_ct']
                degree_ct += temp_res['degree_ct']
                score_detail += temp_res['score_detail']
                break
            else:
                pass

            i += 1

        return {'score':score,'pos_ct':pos_ct,'neg_ct':neg_ct,'deny_ct':deny_ct,
                'degree_ct':degree_ct,'score_detail':score_detail}

    def cal_phrase(self,sub_s,start,after_words):
        '''搜索并计算情感短语得分'''
        tmp,pos_ct,neg_ct= 0,0,0
        tmp_detail = ''
        end = start+1  # end是下一个位置，如果不存在情感短语结构，则下一个位置为start+1
        after_index = min(start+self.phrase_size,len(sub_s))
        for j in range(start,after_index):
            if sub_s[j] in after_words:
                tmp = tmp + float(Lbsa.pre_verb_dict[sub_s[start]])*float(after_words[sub_s[j]])
                tmp_detail = sub_s[start]+sub_s[j]+'; '
                end = j
                break
            else:
                pass
        if tmp>0:
            pos_ct = 1
        if tmp<0:
            neg_ct = 1

        return {'tmp':tmp,'end':end,'pos_ct':pos_ct,'neg_ct':neg_ct,'tmp_detail':tmp_detail}

    def cal_face(self, doc):
        '''计算一篇文档中的表情得分'''
        face_score = 0
        face_detail = ''
        pos_face_ct,neg_face_ct= 0,0
        for i in range(len(doc)):
            if doc[i] in self.face_dict:
                tmp = float(self.face_dict[doc[i]])
                face_score += tmp
                face_detail += doc[i]+' '
                if tmp>0:
                    pos_face_ct += 1
                if tmp<0:
                    neg_face_ct += 1
        if (pos_face_ct+neg_face_ct)>0:
            face_score = face_score/(pos_face_ct+neg_face_ct)
        # if face_score!=0:
        #     print face_detail,face_score
        return {'face_score':face_score, 'pos_face_ct':pos_face_ct, 'neg_face_ct':neg_face_ct}


    def cal_sentence(self,sentence):
        '''计算一个完整句子的规则情感特征'''
        sentence_dict = {'score':0,'pos_ct':0,'neg_ct':0,'pos_sub':0,'neg_sub':0,
                         'deny_ct':0,'degree_ct':0,'score_detail':''}
        sub_sentence_list = tools.cut_sentence(sentence,self.sub_puncs_list)
        sentence_dict['sub_ct'] = len(sub_sentence_list)      #子句数量
        for x in sub_sentence_list:
            sub_res = self.cal_subsent(x)
            sentence_dict['score'] += sub_res['score']
            sentence_dict['pos_ct'] += sub_res['pos_ct']
            sentence_dict['neg_ct'] += sub_res['neg_ct']
            sentence_dict['degree_ct'] += sub_res['degree_ct']
            sentence_dict['score_detail'] += sub_res['score_detail']
            if sub_res['score']>0:
                sentence_dict['pos_sub']+=1
            if sub_res['score']<0:
                sentence_dict['neg_sub']+=1

        return sentence_dict


    def cal_document(self,doc,normalize_opt):
        '''计算一整篇文档的规则情感特征'''
        document_dict = {
            'score':0,
            'pos_ct':0,
            'neg_ct':0,
            'pos_sub':0,
            'neg_sub':0,
            'deny_ct':0,
            'sub_ct':0,
            'degree_ct':0,
            'score_detail':''
        }
        sentence_list = tools.cut_sentence(doc,self.puncs_list)
        document_dict['doc_len'] = tools.cal_len(doc)
        document_dict['word_num'] = len(doc)
        document_dict['sent_num'] = len(sentence_list)          #句子数量
        for x in sentence_list:
            sub_res = self.cal_sentence(x)
            document_dict['score'] += sub_res['score']
            document_dict['pos_ct'] += sub_res['pos_ct']
            document_dict['neg_ct'] += sub_res['neg_ct']
            document_dict['pos_sub'] += sub_res['pos_sub']
            document_dict['neg_sub'] += sub_res['neg_sub']
            document_dict['sub_ct'] += sub_res['sub_ct']
            document_dict['degree_ct'] += sub_res['degree_ct']
            document_dict['score_detail'] += sub_res['score_detail']

        if normalize_opt=='senti_word_num':
            senti_ct = document_dict['pos_ct']+document_dict['neg_ct']
            if senti_ct>0:
                document_dict['score'] = document_dict['score']/senti_ct
        if normalize_opt=='sent_num':
            if document_dict['sent_num']>0:
                document_dict['score'] = document_dict['score']/document_dict['sent_num']
        if normalize_opt=='subsent_num':
            if document_dict['sub_ct']>0:
                document_dict['score'] = document_dict['score']/document_dict['sub_ct']
        if normalize_opt=='word_num':
            if document_dict['word_num']>0:
                document_dict['score'] = document_dict['score']/document_dict['word_num']
        if normalize_opt=='none':
            pass
        # 表情得分加入
        doc_face_dict = self.cal_face(doc)
        document_dict.update(doc_face_dict)
        document_dict['final_score'] = document_dict['score']+document_dict['face_score']
#        if document_dict['sent_len']!=0:
#            document_dict['final_score'] = tools.normalize_score(document_dict['sent_len'],document_dict['final_score'])
#        print document_dict['score_detail']
        return document_dict


