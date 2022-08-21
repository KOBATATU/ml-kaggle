# 機械学習


```
docker-compose run
```

## how to
validationと各種パラメータを宣言することで学習できるようにした
```python

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




if __name__ == "__main__":
    # seed average
    models = [
        *[LgbmModel(seed_average={"random_state":i}) for i in range(2)],
              ]

    from sklearn.datasets import load_iris
    from sklearn.model_selection import  train_test_split
    from sklearn.metrics import f1_score

    with timer("データ処理"):
        iris = load_iris()
        data = iris.data
        target = iris.target
        train_x,valid_x,train_y,valid_y = train_test_split(data,target,stratify=target,random_state=2)

        train_x = pd.DataFrame(train_x,columns=iris.feature_names)
        valid_x = pd.DataFrame(valid_x,columns=iris.feature_names)

    with timer("モデルを推論"):
        preds = 0
        #ここらをstackモデルに変更
        for i,model in enumerate(models):
            oof_pred,pred = model.predict(train_x,valid_x,train_y)
            preds += pred

        LOGGER.info(f1_score(np.argmax(pred,axis=1),valid_y,average="weighted"))
        LOGGER.info(f1_score(np.argmax(preds / len(models),axis=1) ,valid_y,average="weighted")) #seed average
```
