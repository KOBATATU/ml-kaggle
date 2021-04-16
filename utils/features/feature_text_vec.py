import hashlib
import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim.models import FastText

from utils.path_setting import *

from utils.features.base import BaseFeatureMixin

'''
fasttextしようとした時，以下のように記載

import pandas as pd
import gensim
from utils.features.base import BaseFeatureMixin
from utils.features.feature_text import TextFeatureMixin
from utils.features.feature_text_vec import SWEMMixin
from utils.logger import timer, LOGGER

class F2VFeature(SWEMMixin,TextFeatureMixin,BaseFeatureMixin):
    def __init__(self,w2v_params = None):
        super(F2VFeature,self).__init__(encoding_word="F2V")
    def create_features(self, train_df,test_df,df):
        output_df = pd.DataFrame()
        //  googlenews_w2v = gensim.models.KeyedVectors.load_word2vec_format("./data/processed/GoogleNews-vectors-negative300.bin",binary=True)

        for col in use_columns:
            LOGGER.info("PROCESS F2v {}".format(col))
            document = self.preprocess(df[col])
            document = [docs.split(" ") for docs in document.to_list()]
            vector_df = self.create_f2v_vector(document=document,col=col)
            //vector_df = self.create_f2v_vector(document=document,f2v = googlenews_w2v,col=col) 既存のモデルを使用

            output_df = pd.concat([output_df,vector_df],axis=1)
            LOGGER.info("SHAPE: {}".format(output_df.shape))
        return output_df
        
        
Featureを連続的にconcatしないとき．

class F2VFeature(SWEMMixin,TextFeatureMixin,BaseFeatureMixin):
    def __init__(self,w2v_params = None):
        super(F2VFeature,self).__init__(encoding_word="F2V")
        
    @classmethod
    def make(self, df):
        output_df = pd.DataFrame()
        //  googlenews_w2v = gensim.models.KeyedVectors.load_word2vec_format("./data/processed/GoogleNews-vectors-negative300.bin",binary=True)

        for col in use_columns:
            LOGGER.info("PROCESS F2v {}".format(col))
            document = self.preprocess(df[col])
            document = [docs.split(" ") for docs in document.to_list()]
            vector_df = self.create_f2v_vector(document=document,col=col)
            //vector_df = self.create_f2v_vector(document=document,f2v = googlenews_w2v,col=col) 既存のモデルを使用

            output_df = pd.concat([output_df,vector_df],axis=1)
            LOGGER.info("SHAPE: {}".format(output_df.shape))
            
        self.perform_create(df)
        
        return output_df

vector_df = F2VFeature().make(df)


'''

def hashfxn(x):
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)



class WordVector(object):

    def _get_path(self):
        if self.col is None:
            self.col = "untitled"
        path = os.path.join(PROCESSED_ROOT, f"{self.col}.model")
        return path

    def _update_params(self,params):
        # default params
        self.params = {
            "size": 40,
            "iter": 30,
            "workers": 1,
            "hashfxn": hashfxn
        }

        if params is not None:
            for key in params.keys():
                self.params[key] = params[key]


    def train(self, token,params,method="w2v"):
        #pathがあればモデルをロードする．
        path = self._get_path()
        if os.path.exists(path):
            self.load(path,method)

        else:
            self._update_params(params)
            if method=="w2v":
                self.model = word2vec.Word2Vec(token, **self.params)
            elif method == "f2v":
                self.model = FastText(token, **self.params)

            self.save(path)


    def save(self, path):
        print("SAVE!!")
        self.model.save(path)

    def load(self, path,method):
        if method == "w2v":
            self.model = word2vec.Word2Vec.load(path)
        elif method == "f2v":
            self.model = FastText.load(path)


class SWEMMixin(WordVector):
    '''
    simple word Embedding models
    '''



    def get_word_embeddings(self, text):

        vectors = []

        for word in text:
            if word in self.vocab:
                vectors.append(self.model[word])
            else:
                vectors.append(np.zeros(self.embedding_dim))

        # 何もない時の処理
        if len(vectors) == 0:
            vectors.append(np.zeros(self.embedding_dim))
        return np.array(vectors)

    def average_pooling(self, sentence):
        '''
        sentence:list
        '''
        result = []
        for text in sentence:
            word_embeddings = self.get_word_embeddings(text)
            average_vectors = np.mean(word_embeddings, axis=0)
            result.append(average_vectors)

        columns = [f"{self.col}_average_vector_{i}" for i in range(self.embedding_dim)]
        result = pd.DataFrame(result, columns=columns)


        return result


    def create_w2v_vector(self,document,w2v=None,col=None,w2v_params = None):
        '''

        :param document: [["hoge","hogehoge"],["aa","aaff"]]
        :return: Vector DataFrame

        '''

        self.col = "w2v_"+col #w2vでvectorを作っているため，最初にw2vを入れる．
        if w2v is not  None:
            self.model = w2v  #学習モデルがある場合それを取得する．
        else:
            self.train(document,params=w2v_params,method="w2v")

        self.vocab = set(self.model.wv.vocab.keys()) #self.model = word2vecモデル
        self.embedding_dim = self.model.wv.vector_size


        return self.average_pooling(document)

    def create_f2v_vector(self,document,f2v=None,col=None,f2v_params = None):
        '''

        :param document: [["hoge","hogehoge"],["aa","aaff"]]
        :return: Vector DataFrame

        '''

        self.col = "f2v_" + col  # f2vでvectorを作っているため，最初にf2vを入れる．
        if f2v is not None:
            self.model = f2v  # 学習モデルがある場合それを取得する．(path指定のモデルがある場合)
        else:
            self.train(document, params=f2v_params,method="f2v")

        self.vocab = set(self.model.wv.vocab.keys())  # self.model = word2vecモデル
        self.embedding_dim = self.model.wv.vector_size

        return self.average_pooling(document)





