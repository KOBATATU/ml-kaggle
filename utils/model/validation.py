from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold

class FoldValidation():
    def __init__(self,fold_num = 4,random_state = 1234,shuffle_flg = True,fold_type=None,split_time_list = None):
        self.fold_num = fold_num
        self.random_state = random_state
        self.shuffle_flg = shuffle_flg
        self.fold_type = fold_type
        self.split_time_list = split_time_list

    def make_splits(self,train,y,unique_id_col = None,time_col = None,groups = None):
        '''
        valid_typeは以下の中
        - Stratified
        - Kfold
        - TimeSplit
        - GroupKfold
        '''
        if self.fold_type == "TimeSplit":
            assert (time_col is not None)
        if self.fold_type == "GroupKfold":
            assert (groups is not None)

        if self.fold_type == "Stratified":
            self.folds = StratifiedKFold(
                n_splits=self.fold_num,
                shuffle=self.shuffle_flg,
                random_state=self.random_state
            )
            self.split_index_list = [(train_ind,val_ind) for train_ind,val_ind in self.folds.split(train,y)]

        elif self.fold_type == "Kfold":
            self.folds = KFold(
                n_splits=self.fold_num,
                shuffle=self.shuffle_flg,
                random_state=self.random_state
            )
            self.split_index_list = [(train_ind,val_ind) for train_ind,val_ind in self.folds.split(train,y)]

        elif self.fold_type == "TimeSplit":
            self.split_index_list = []
            for split_time in self.split_time_list:
                trn_idx = train[train[time_col].dt.date < split_time].index
                val_idx = train[train[time_col].dt.date >= split_time].index
                self.split_index_list.append((trn_idx, val_idx))
        else:
            raise ("Not Implemented Error")

    def get_split_index(self,train,y,unique_id_col = None,time_col = None,groups = None):
        self.make_splits(train,y,unique_id_col = None,time_col = None,groups = None)
        return self.split_index_list
