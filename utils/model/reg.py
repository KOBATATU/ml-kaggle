import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

from utils.model.base import BaseMixin
from utils.path_setting import *


class BasicRegMixin(BaseMixin):

    def __init__(self,path):
        self.path = os.path.join(PREDICT_FOR_TEST_DIR,path)
        os.makedirs(self.path, exist_ok=True)


    def _metric(self, pred, y_valid, i):
        mae = mean_absolute_error(y_valid, pred)
        rmse = mean_squared_error(y_valid, pred)**0.5
        msle = mean_squared_log_error(y_valid, np.where(pred <= 0, 0, pred))

        metric_df = pd.DataFrame([[mae, rmse, msle]], columns=["mae", "RMSE", "Root Mean Squared Log Error"])
        metric_df.to_csv(os.path.join(self.path, "_" + str(i) + "_metric.csv"), index=False)


    def fit(self,train_df:pd.DataFrame,y:np.array,target_split=None):


        columns = train_df.columns
        X = train_df.values

        split_idx = self._get_split_idx(X,y,target_split)

        models = []
        oof_pred = np.zeros(len(y), dtype=np.float)
        for i, (idx_train, idx_valid) in enumerate(split_idx):
            # training data を trian/valid に分割
            x_train, y_train = X[idx_train], y[idx_train]
            x_valid, y_valid = X[idx_valid], y[idx_valid]
            model = self._learning(x_train,y_train,x_valid,y_valid)

            pred_i = model.predict(x_valid)
            oof_pred[idx_valid] = pred_i

            self._metric(pred_i, y_valid, i)  # trainデータを分割して，予測modelと評価データとで比較．
            models.append(model)

        self.feature_importances(models,columns)

        return oof_pred, models

    #クラス分類と回帰モデルではpredictが違う．
    def predict(self, train_df:pd.DataFrame,test_df:pd.DataFrame,y:np.array,target_split=None):

        if train_df.shape[1] != test_df.shape[1]:
            raise ValueError("trainとtestのカラム数が異なります")

        oof_pred,  models = self.fit(train_df,y,target_split)

        pred = [m.predict(test_df.values) for m in models]
        pred = np.mean(pred, axis=0)

        self.perform_create(oof_pred,pred)



        return oof_pred, pred


#設定は以下の通り．
# class CatModel(BoostingMixin,BasicRegMixin):
#     def __init__(self,add_params=None,seed_average={"random_state":0}):
#
#         self.random_state = seed_average["random_state"]
#         super(CatModel,self).__init__(path = "Cat_"+str(self.random_state)) #self.pathが定義
#
#         self.params = {
#             # 目的関数. これの意味で最小となるようなパラメータを探します.
#             'objective': 'RMSE',
#
#             # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、
#             # がそれだけ木を作るため学習に時間がかかります
#             'learning_rate': 0.03,
#
#             # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
#             'n_estimators': 10000,
#
#             'random_state':self.random_state ,
#         }
#
#
#         self.validation = FoldValidation(
#         fold_num =  5,
#         random_state=self.random_state ,
#         shuffle_flg=True,
#         fold_type="Stratified"
#
#         )
#
#         self.model = CatBoostRegressor

