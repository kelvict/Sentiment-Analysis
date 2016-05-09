# -*- coding: utf-8 -*-

from __future__ import division
import os
import re
import random
import math
import subprocess
import copy
import tools
from Lbsa import Lbsa

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

def gen_N_gram(doc_str_list,ngram='uni'):
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
