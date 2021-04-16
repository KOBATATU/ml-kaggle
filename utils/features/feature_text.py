import re,string
import os,sys
import hashlib
import MeCab
import numpy as np
import pandas as pd
import texthero as hero
import texthero.preprocessing as heropreprocess
import pycld2 as cld2

'''

from utils.features.base import BaseFeatureMixin
from utils.features.feature_text import TextFeatureMixin


class BasicTextFeature(TextFeatureMixin,BaseFeatureMixin):
    def __init__(self):
        super(BasicTextFeature,self).__init__(encoding_word = "BasicTextFeature")

    def create_features(self,train_df,test_df,df):
        df = self.create_text_count_feature(df,train_df,test_df,use_columns,method_list=["word_counts","letter_counts","digits_counts"])
        return df

'''
class TextFeatureMixin():

    def preprocess(self,document,pipeline = None,add_preprocess=None):
        """
         Return a list contaning all the methods used in the default cleaning
         pipeline.
         Return a list with the following functions:
          1. :meth:`texthero.preprocessing.fillna`
          2. :meth:`texthero.preprocessing.lowercase`
          3. :meth:`texthero.preprocessing.remove_digits`
          4. :meth:`texthero.preprocessing.remove_html_tags`
          5. :meth:`texthero.preprocessing.remove_punctuation`
          6. :meth:`texthero.preprocessing.remove_diacritics`
          7. :meth:`texthero.preprocessing.remove_stopwords`
          8. :meth:`texthero.preprocessing.remove_whitespace`

         """
        if pipeline is not None:
            pipeline = heropreprocess.get_default_pipeline()
        if add_preprocess is not None:
            for preprocess in add_preprocess:
                pipeline.append(preprocess)



        clean_text = hero.clean(document, pipeline=pipeline)
        return clean_text


    def word_counts(self,document):
        number_of_words = [len(docs.split()) for docs in document] #word数
        return number_of_words

    def letter_counts(self,document):
        number_of_letter=  [len(docs) for docs in document] #letters数
        return number_of_letter

    # 使われている言語を判定
    def judge_language(self, document):
        language = [cld2.detect(docs)[2][0][1] for docs in document]
        return language


    # basicではないが時たま使えるであろうfeature
    #作られるfeatureが多いためcreateにしている．
    def create_onehot_alhabet(self, col, document):
        import string
        alhabet_dict = {}
        for letters in string.ascii_letters[:26]:
            cols_name = f"{col}_number_of_alpfabet_{letters}"
            cols_alpfabet = [docs.lower().count(letters) for docs in document]
            alhabet_dict[cols_name] = cols_alpfabet

        return pd.DataFrame(alhabet_dict)

    def create_onehot_number(self,col,document):
        digits_dict = {}
        for letters in string.digits:
            cols_name = f"{col}_number_of_digits_{letters}"
            cols_alpfabet = [docs.lower().count(letters) for docs in document]
            digits_dict[cols_name] = cols_alpfabet

        return pd.DataFrame(digits_dict)




    def get_text_method(self,text_feature_method):
        if text_feature_method == "word_counts":
            return self.word_counts
        elif text_feature_method == "letter_counts":
            return self.letter_counts
        elif text_feature_method == "alhabet_counts":
            return self.create_onehot_alhabet
        elif text_feature_method == "digits_counts":
            return self.create_onehot_number
        elif text_feature_method == "judge_language":
            return self.judge_language
        else:
            raise ValueError("該当するものがありません")

    def create_text_count_feature(self,df,train_df,test_df,use_columns,method_list):
        output_df = pd.DataFrame()
        for col in use_columns:
            document = self.preprocess(df[col]) #preprocess
            for text_feature_method in method_list:
                method = self.get_text_method(text_feature_method)
                if text_feature_method == "word_counts" or text_feature_method == "letter_counts" or \
                        text_feature_method=="judge_language":
                    output_df[f"{col}_{text_feature_method}"] = method(document)
                else:
                    _other_df = method(col,document)
                    output_df = pd.concat([output_df,_other_df],axis=1)
        return output_df





class Wakati(object):

    def __init__(self, grammer=["名詞"]):
        self.mecab = MeCab.Tagger("-d  /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
        self.grammer = grammer

    def parse(self, sentence):
        '''
        sentence = [
            "hogehoge",
            "hogehoge",
        ]

        result = [
            ["hoge","hoge"],
            ["hoge","hoge"]
        ]
        '''
        result = []

        for text in sentence:
            lines = self.mecab.parse(text)
            lines = lines.split("\n")
            line_result = []

            for line in lines:
                feature = line.split("\t")
                if len(feature) == 2:
                    analyze = feature[1].split(",")
                    gram = analyze[0]
                    if gram in self.grammer:
                        line_result.append(analyze[6])

            result.append(line_result)

        return result



