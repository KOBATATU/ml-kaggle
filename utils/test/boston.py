
import os,sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import lightgbm as lgbm
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

from utils.model.base import BoostingMixin
from utils.model.reg import BasicRegMixin
from utils.model.validation import FoldValidation
from utils.logger import timer,LOGGER

class LgbmModel(BoostingMixin,BasicRegMixin):
    def __init__(self,seed_average={"random_state":0}):

        self.random_state = seed_average["random_state"]
        super(LgbmModel,self).__init__(path = "Lgbm_"+str(self.random_state)) #self.pathが定義

        self.params = {
            'objective': 'rmse',


            'learning_rate': 0.03,

            'n_estimators': 10000,

            'random_state':self.random_state ,
        }


        self.validation = FoldValidation(
        fold_num =  5,
        random_state=self.random_state ,
        shuffle_flg=True,
        fold_type="Kfold"

        )

        self.model = lgbm.LGBMRegressor

if __name__ == "__main__":
    # seed average
    models = [
        *[LgbmModel(seed_average={"random_state":i}) for i in range(2)],
              ]

    from sklearn.datasets import load_boston
    from sklearn.model_selection import  train_test_split
    from sklearn.metrics import mean_squared_error

    with timer("データ処理"):
        boston = load_boston()
        data = boston.data
        target = boston.target
        train_x,valid_x,train_y,valid_y = train_test_split(data,target,random_state=2)

        train_x = pd.DataFrame(train_x,columns=boston.feature_names)
        valid_x = pd.DataFrame(valid_x,columns=boston.feature_names)

    with timer("モデルを推論"):
        preds = 0
        #ここらをstackモデルに変更
        for i,model in enumerate(models):
            oof_pred,pred = model.predict(train_x,valid_x,train_y)
            preds += pred

        LOGGER.info(mean_squared_error(pred,valid_y))
        LOGGER.info(mean_squared_error(preds / len(models),valid_y))
        # INFO - 8.453883012175464
        # INFO - 8.176236572245928



