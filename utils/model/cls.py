import  os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score,log_loss

from utils.model.base import BaseMixin
from utils.path_setting import *


class BasicClfMixin(BaseMixin):
    def __init__(self, path):
        self.path = os.path.join(PREDICT_FOR_TEST_DIR, path)
        os.makedirs(self.path, exist_ok=True)

    def _metric(self, pred, y_valid, i):
        loss = log_loss(y_pred=pred,y_true=y_valid)

        #確率値にする．
        pred = np.argmax(pred,axis=1)
        acc = accuracy_score(y_valid, pred)
        f1_micro = f1_score(y_valid, pred,average="micro")
        f1_weight = f1_score(y_valid, pred,average="weighted")
        metric_df = pd.DataFrame([[acc, f1_micro,f1_weight,loss]], columns=["acc", "Micro_F1","Weight_F1","log_loss"])
        metric_df.to_csv(os.path.join(self.path, "_" + str(i) + "_metric.csv"), index=False)

    def fit(self, train_df: pd.DataFrame, y: np.array, target_split=None):

        columns = train_df.columns
        X = train_df.values
        split_idx = self._get_split_idx(X, y, target_split)

        models = []
        oof_pred = np.zeros((len(y),len(np.unique(y))), dtype=np.float)
        for i, (idx_train, idx_valid) in enumerate(split_idx):
            # training data を trian/valid に分割
            x_train, y_train = X[idx_train], y[idx_train]
            x_valid, y_valid = X[idx_valid], y[idx_valid]
            model = self._learning(x_train, y_train, x_valid, y_valid)

            pred_i = model.predict_proba(x_valid)
            oof_pred[idx_valid] = pred_i

            self._metric(pred_i, y_valid, i)  # trainデータを分割して，予測modelと評価データとで比較．
            models.append(model)

        self.feature_importances(models, columns)

        return oof_pred, models


    def predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame, y: np.array, target_split=None):

        if train_df.shape[1] != test_df.shape[1]:
            raise ValueError("trainとtestのカラム数が異なります")

        oof_pred, models = self.fit(train_df, y, target_split)

        pred = [m.predict_proba(test_df.values) for m in models]
        pred = np.mean(pred, axis=0)
        self.perform_create(oof_pred, pred)
        return oof_pred, pred


# 設定は以下の通り．

# import lightgbm as lgbm
# from utils.model.base import BoostingMixin
# from utils.model.validation import FoldValidation
# from utils.logger import timer,LOGGER
#
#
# class LgbmModel(BoostingMixin, BasicClfMixin):
#     def __init__(self, seed_average={"random_state": 0}):
#         self.random_state = seed_average["random_state"]
#         super(LgbmModel, self).__init__(path="Lgbm_CLS" + str(self.random_state))  # self.pathが定義
#
#         self.params = {
#
#             # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、
#             # がそれだけ木を作るため学習に時間がかかります
#             'learning_rate': 0.03,
#
#             # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
#             'n_estimators': 10000,
#
#             'random_state': self.random_state,
#         }
#
#         self.validation = FoldValidation(
#             fold_num=5,
#             random_state=self.random_state,
#             shuffle_flg=True,
#             fold_type="Stratified"
#
#         )
#
#         self.model = lgbm.LGBMClassifier

