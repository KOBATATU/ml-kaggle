import numpy as np
import pandas as pd

from utils.path_setting import *
from utils.logger import timer,LOGGER

'''
import copy

#output_trainとoutput_testは作成された特徴量のみが返却される．

for FeatureObject in [BasicAggregation,FrequencyMixin]:
    output_train,output_test = FeatureObject().make(train_df,test_df)
    train_df = pd.concat([train_df,output_train],axis=1)
    test_df = pd.concat([test_df,output_test],axis=1)

'''

class BaseFeatureMixin(object):

    def __init__(self,encoding_word):
        self.encoding_word = encoding_word
        self.path =  os.path.join(PROCESSED_ROOT,encoding_word + ".csv")

        

    def create_use_features_df(self,df):
        use_feature = [
            c for c in df.columns
            if c not in self.EXCEPT_COLUMNS
        ]

        df = df[use_feature]

        return df

    def create_features(self,train_df,test_df,df):
        '''
        必ずオーバーライドする．
        '''
        pass

    def peform_create(self,df,encoding_word):
        '''
        作成された特徴量を保存．
        特殊な保存を行う場合はオーバーライド
        '''
        df.to_csv(self.path,index=False)

    def LoggerCheck(self,train_df,test_df,df):
        LOGGER.info("--------------------------------------")
        LOGGER.info(f"{self.encoding_word} OBJECT TRAIN SHAPE : {train_df.shape}")
        LOGGER.info(f"{self.encoding_word} OBJECT TEST SHAPE : {test_df.shape}")
        LOGGER.info(f"{self.encoding_word} OBJECT DF SHAPE : {df.shape}")



    def make(self,train_df,test_df):
        self.EXCEPT_COLUMNS = train_df.columns
        n_train = len(train_df)

        if os.path.exists(self.path):
            #既に特徴量が作られている場合はread_csvで読み込む．
            df = pd.read_csv(self.path)

        else:
            df = pd.concat([train_df, test_df]).reset_index(drop=True)
            df = self.create_features(train_df,test_df,df)#特徴量を作成
            df = self.create_use_features_df(df) #不必要な特徴量を消去し必要なdfだけを返却
            self.peform_create(df,self.encoding_word) #作成された特徴量は保存

        train_df = df.iloc[:n_train].reset_index(drop=True)
        test_df = df.iloc[n_train:].reset_index(drop=True)
        self.LoggerCheck(train_df,test_df,df)


        return train_df,test_df


def feature_debug(df,idx = 30,reverse = None):
    LOGGER.info(f"---------------COLUMNS EXAMPLE {idx}-----------------------")
    LOGGER.info(df.columns[:idx])
    if reverse is not None:
        LOGGER.info(f"-----------------COLUMNS EXAMPLE REVERSE {idx}----------------")
        LOGGER.info(df.columns[-30:])
