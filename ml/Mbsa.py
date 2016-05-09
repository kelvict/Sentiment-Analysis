# -*- coding: utf-8 -*-
"""
@date: 2016.5.9
@author: JieJ

"""
import os


class Mbsa(object):
    def __init__(self, token_gram = 'uni', token_fs_opt = 0, token_fs_method = 'WLLR', token_fs_num = -1, token_df = 2,
        pos_gram='none', pos_fs_opt = 0, pos_fs_method = 'WLLR', pos_fs_num = -1, pos_df = 2,
        character_gram = (2,4), character_fs_opt = 1, character_fs_method = 'WLLR', character_fs_num = 3000, character_df = 2,
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

    def gen_doc_terms_list(self, token_fname_list, pos_fname_list, class_fname_list, nfolds=0):
        doc_str_token, doc_class_list = pytc.read_annotated_data([train_dir + \
            os.sep + x for x in token_fname_list], class_fname_list)
        doc_token_list = pytc.gen_N_gram(doc_str_token,self.token_gram)
        token_set = pytc.get_term_set(doc_token_list)

        doc_str_pos = []
        if self.pos_gram != 'none':
            doc_str_pos, doc_class_list = pytc.read_annotated_data([train_dir + os.sep + x \
                for x in pos_fname_list], class_fname_list)
            doc_pos_list = pytc.gen_N_gram(doc_str_pos,self.pos_gram)
            pos_set = pytc.get_term_set(doc_pos_list)

        if self.character_gram == '1':
            doc_character_list = pytc.gen_character_ngram_list(doc_str_token)
            character_set = pytc.get_term_set(doc_character_list)

        doc_terms_list = pytc.get_joint_sets(doc_token_list, doc_pos_list)
        doc_terms_list = pytc.get_joint_sets(doc_terms_list, doc_character_list)

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

        term_set = token_set + pos_set
        term_set = term_set + character_set

        return doc_str_token, doc_terms_list, term_set
