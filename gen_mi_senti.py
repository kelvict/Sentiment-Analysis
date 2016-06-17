# -*- coding: utf-8 -*-
"""
根据弱标注语料生成通用词典

"""

from __future__ import division
import os
import math

def get_term_set(fenci_lines):

    term_set = set()
    for line in fenci_lines:
        line_set = set(line.strip().split())
        term_set.update(line_set)

    return list(term_set)


def mi_sentidict(pos_fenci_lines, neg_fenci_lines):
    fenci_lines = pos_fenci_lines + neg_fenci_lines
    term_set = get_term_set(fenci_lines)

    term_pos_freq = {}.fromkeys(term_set, 0)
    term_neg_freq = {}.fromkeys(term_set, 0)

    freq_pos, freq_neg = 0, 0

    for line in pos_fenci_lines:
        lst = line.strip().split()
        for term in lst:
            term_pos_freq[term] += 1
            freq_pos += 1

    for line in neg_fenci_lines:
        lst = line.strip().split()
        for term in lst:
            term_neg_freq[term] += 1
            freq_neg += 1

    mi_senti_dict = {}

    for term in term_set:
        if term in term_pos_freq and term in term_neg_freq:
            tmp = ((term_pos_freq[term] + 1) * freq_neg) / ((term_neg_freq[term] + 1) * freq_pos)
            mi_senti_dict[term] = round(math.log(tmp, 2), 4)

    return mi_senti_dict
