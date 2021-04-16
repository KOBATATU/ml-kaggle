import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

from utils.path_setting import *

'''

class StackModel():
    def __init__(self,model_dict):
        self.model_dict = model_dict
    def predict(self,train_df,y,test_df,):
        for model_name in self.model_dict:
            _model_dict = self.model_dict[model_name]
            preds = [model.predict(train_df,y,test_df) for model in _model_dict["list"]]
            preds = np.mean(preds,axis=1) * _model_dict["weight"]

'''

class BaseMixin(object):
    '''
    このクラスはクラス分類でも回帰モデルでも変わらないメソッドを入れる．
    '''
    #boostingモデル用にfeature importanceを継承する
    def feature_importances(self, models: list, columns):
        pass

    #予測したものを保存
    def perform_create(self,oof_pred,pred):
        pd.DataFrame(oof_pred).to_csv(os.path.join(self.path,"_oof_pred.csv"),index=False)
        pd.DataFrame(pred).to_csv(os.path.join(self.path,"_pred.csv"),index=False)

    #validation
    def _get_split_idx(self, X, y, target_split):
        if target_split is not None:
            split_idx = self.validation.get_split_index(X, target_split)  # targetが自動でvalidationの基準に
        else:
            split_idx = self.validation.get_split_index(X, y)  # targetが自動でvalidationの基準に

        return split_idx


    # boostingアルゴリズムはここが違うので，overrideして上書きする．
    def _learning(self, x_train, y_train, x_valid, y_valid):
        model = self.model(**self.params)
        model.fit(x_train, y_train)
        return model



class BoostingMixin(object):

    def _learning(self, x_train, y_train,x_valid,y_valid):
        model = self.model(**self.params)
        model.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    early_stopping_rounds=100,
                    verbose=1000)
        return model

    def feature_importances(self, models, columns):
        df = pd.DataFrame()
        for i, model in enumerate(models):
            feature_importance = pd.DataFrame()
            feature_importance["feature_importance"] = model.feature_importances_
            feature_importance["columns"] = columns
            df = pd.concat([df, feature_importance], axis=0)
        order = df.groupby('columns')[['feature_importance']].mean().sort_values(
            'feature_importance',
            ascending=False).index[:40]

        fig = plt.figure(figsize=(len(columns) * .1, 7))
        ax = fig.add_subplot(111)
        sns.barplot(data=df, y='columns', x='feature_importance', order=order, ax=ax)
        ax.tick_params(axis='x', rotation=90)
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, "_feature_importances.png"), dpi=150)#LGBM_0_feature_importtance.png



