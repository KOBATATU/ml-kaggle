import os,sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import lightgbm as lgbm
from catboost import CatBoostClassifier

from utils.model.base import BoostingMixin
from utils.model.validation import FoldValidation
from utils.model.cls import BasicClfMixin
from utils.logger import timer,LOGGER


#BoostingMixinとClfMixinを継承
class LgbmModel(BoostingMixin, BasicClfMixin):
    def __init__(self, seed_average={"random_state": 0}):
        self.random_state = seed_average["random_state"]
        super(LgbmModel, self).__init__(path="Lgbm_CLS" + str(self.random_state))  # self.pathが定義

        self.params = {
            'learning_rate': 0.03,
            'n_estimators': 10000,
            'random_state': self.random_state,
        }

        self.validation = FoldValidation(
            fold_num=5,
            random_state=self.random_state,
            shuffle_flg=True,
            fold_type="Stratified"

        )

        self.model = lgbm.LGBMClassifier

#BoostingMixinとClfMixinを継承
class CatModel(BoostingMixin, BasicClfMixin):
    def __init__(self, seed_average={"random_state": 0}):
        self.random_state = seed_average["random_state"]
        super(CatModel, self).__init__(path="CAT_CLS" + str(self.random_state))  # self.pathが定義

        self.params = {
            'learning_rate': 0.03,
            'n_estimators': 10000,
            'random_state': self.random_state,
        }

        self.validation = FoldValidation(
            fold_num=5,
            random_state=self.random_state,
            shuffle_flg=True,
            fold_type="Stratified"

        )

        self.model = CatBoostClassifier



if __name__ == "__main__":
    # seed average
    models = [
        *[LgbmModel(seed_average={"random_state":i}) for i in range(2)],
        *[CatModel(seed_average={"random_state":i}) for i in range(2)]
              ]

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import  train_test_split
    from sklearn.metrics import f1_score

    with timer("データ処理"):
        cancer = load_breast_cancer()
        data = cancer.data
        target = cancer.target
        train_x,valid_x,train_y,valid_y = train_test_split(data,target,stratify=target,random_state=2)

        train_x = pd.DataFrame(train_x,columns=cancer.feature_names)
        valid_x = pd.DataFrame(valid_x,columns=cancer.feature_names)

    with timer("モデルを推論"):
        preds = 0
        #ここらをstackモデルに変更
        for i,model in enumerate(models):
            oof_pred,pred = model.predict(train_x,valid_x,train_y)
            preds += pred

        LOGGER.info(f1_score(np.argmax(pred,axis=1),valid_y,average="binary"))
        LOGGER.info(f1_score(np.argmax(preds / len(models),axis=1) ,valid_y,average="binary")) #seed average

