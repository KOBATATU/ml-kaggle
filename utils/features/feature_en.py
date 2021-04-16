import numpy as np
import pandas as pd

'''
from utils.features.base import BaseFeatureMixin
from utils.features.feature_en import GroupAggregationMixin

class BasicAggregation(GroupAggregationMixin,BaseFeatureMixin):
    def __init__(self):
        super(BasicAggregation,self).__init__(encoding_word = "Aggregation")

    def create_features(self,train_df,test_df,df):
        df= self.transform(train_df,test_df,stats,agg_features,group_features, merge=True)
        return df

'''

class GroupAggregationMixin(object):
 

    def _aggregate(self, train_df,test_df, merge=None):
    
        aggs_df = pd.DataFrame()
        for group in self.group_features:
            for agg in self.agg_features:
                stats_df = train_df.groupby(group).agg({agg: self.stats})
                # stats_df.fillna(0,inplace=True)
                stats_columns = pd.Index(
                    ["agg_key={}_".format(group) + "_fe=" + col[0] + "_" + col[1] for col in stats_df.columns.tolist()])
                stats_df.columns = stats_columns
                
                if merge is not None:
                    stats_df.reset_index(inplace=True)
                    train_df = pd.merge(train_df, stats_df, on=group, how="left")
                    test_df = pd.merge(test_df, stats_df, on=group, how="left")

                else:
                    return stats_df

        return train_df,test_df

    def transform(self, train_df,test_df, stats,agg_features,group_features,merge=True):
        """
        :param stats: aggregation
        :param agg_features: 計算させたいやつ
        :param group_features: groupでまとめる
        """
        self.stats = stats
        self.agg_features = agg_features
        self.group_features = group_features

        train_df,test_df = self._aggregate(train_df,test_df, merge=merge)
        df = pd.concat([train_df,test_df]).reset_index(drop=True)
        return df


'''
from utils.features.base import BaseFeatureMixin
from utils.features.feature_en import FrequencyMixin

class BasicFrequency(FrequencyMixin,BaseFeatureMixin):
    def __init__(self):
        super(BasicFrequency,self).__init__(encoding_word = "Aggregation")

    def create_features(self,train_df,test_df,df):
        use_columns = [

        ]
        df= self.create_frequency_encoding(train_df,test_df,use_columns)
        return df

class BasicCategoryEncoding(OneHotMixin,FrequencyMixin,BaseFeatureMixin):
      def create_features(self,train_df,test_df,df):
        use_columns = [

        ]
        freq_df= self.create_frequency_encoding(train_df,df,use_columns)
        onehot_df = self.create_one_hot_encoding(train_df,df,use_columns,vc=20)

        df = pd.concat([freq_df,onehot_df],axis=1)

        return df


'''
class FrequencyMixin(object):

    def create_frequency_encoding(self,train_df,df,use_columns):
        out_df = pd.DataFrame()
        for column in use_columns:
            vc = train_df[column].value_counts() #countベースのカテゴリを定量化
            out_df[column] = df[column].map(vc)

        return out_df.add_prefix('CE_')

class OneHotMixin(object):
    def create_one_hot_encoding(self,train_df,input_df,use_columns,**kwargs):
        out_df = pd.DataFrame()
        for column in use_columns:

            # あまり巨大な行列にならないよう,出現回数が n 回を下回るカテゴリは考慮しない
            vc = train_df[column].value_counts()
            if "vc" in kwargs.keys():
                vc = vc[vc > kwargs["vc"]]

            # 明示的に catgories を指定して, input_df によらず列の大きさが等しくなるようにする
            cat = pd.Categorical(input_df[column], categories=vc.index)

            # このタイミングで one-hot 化
            out_i = pd.get_dummies(cat)
            # column が Catgory 型として認識されているので list にして解除する (こうしないと concat でエラーになる)
            out_i.columns = out_i.columns.tolist()
            out_i = out_i.add_prefix(f'OneHot_{column}=')
            out_df = pd.concat([out_df, out_i], axis=1)
        return out_df


class SinCosMixin():

    def create_sin_cos_features(self, df,col,period):
        output_df = pd.DataFrame()
        '''
        :param df:
        :param col:
        :param period: month = 12 day = 365 hour = 24 minute = 60
        :return:
        '''

        output_df['{}_sin'.format(col)] = np.sin(2 * np.pi * df[col]/period)
        output_df['{}_cos'.format(col)] = np.cos(2 * np.pi * df[col] / period)
        new_cols = ['{}_sin'.format(col),'{}_cos'.format(col)]
        return output_df,new_cols