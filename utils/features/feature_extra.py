import colorsys
import numpy as np
import pandas as pd

'''
色々なfeatureを作るときに便利なクラス．

'''


class ExtraFeatureMixin():

    def color_rgb_to_hsv(self,R,G,B):
        H, S, V = colorsys.rgb_to_hsv(R, G, B)
        return  H, S, V
    def color_rgb_to_hls(self,R,G,B):
        H,L,S = colorsys.rgb_to_hls(R, G, B)
        return H,L,S

    def color_rgb_to_yiq(self,R,G,B):
        Y,I,Q = colorsys.rgb_to_yiq(R, G, B)
        return Y,I,Q

    def create_color_rgb_to_other(self,R,G,B,**kwargs):
        output_df = pd.DataFrame()
        percentage = 1
        #percentageがある場合
        if "percentage" in kwargs["percentage"]:
            percentage = kwargs["percentage"]


        for change in ["hsv", "hls", "yiq"]:
            eps = 1e-10
            result_list_r = []
            result_list_g = []
            result_list_b = []
            for r, g, b in zip(R + eps, G + eps, B + eps):
                if change == "hsv":
                    H, S, V = self.color_rgb_to_hsv(r, g, b)
                elif change == "hls":
                    # hls
                    H,S,V = self.color_rgb_to_hls(r,g,b)
                elif change == "yiq":
                    # yiq
                    H, S, V = self.color_rgb_to_yiq(r, g, b)

                result_list_r.append(H)
                result_list_g.append(S)
                result_list_b.append(V)

            output_df[f"color_rgb_to_{change}_r"] = percentage * np.array(result_list_r)
            output_df[f"color_rgb_to_{change}_g"] = percentage * np.array(result_list_g)
            output_df[f"color_rgb_to_{change}_b"] = percentage * np.array(result_list_b)

        return output_df
    # 使わないと思う．．．．
    def create_cross_tab(self,train_df,test_df,df_dict,key):
        '''

        :param train_df:
        :param test_df:
        :param df:
        :param df_dict: {CROSS:[PATH:,NAME:"Principal",USE_COLUMNS:[],VC:20]",}
        :param key:集約情報．
        :return:
        '''

        for _df_dict in df_dict:

            try:

                name = _df_dict["NAME"]
                use_columns = _df_dict["USE_COLUMNS"]
                v = _df_dict["VC"]
                _df = pd.read_csv(_df_dict["PATH"])

            except:
                raise ValueError(
                    '''
                    指定されたDictのValueが存在しません.
                     df_dict: {CROSS:[PATH:,NAME:"Principal",USE_COLUMNS:[],VC:20]"}
                     で指定してください．
                    '''
                )

            for col in use_columns:
                vc = _df[col].value_counts()
                # 出現回数30以上に絞る
                use_names = vc[vc >= v].index

                # isin で v 回以上でてくるようなレコードに絞り込んでから corsstab を行なう
                idx = _df[col].isin(use_names)
                _use_df = _df[idx].reset_index(drop=True)

                cross_tab_df = pd.crosstab(_use_df[key], _use_df[col]).reset_index()
                #特定のkey情報以外の列の名前を変更
                cross_tab_key_df = cross_tab_df[key]
                cross_tab_df.drop([key],axis=1,inplace=True)
                columns = ["CrossTab_"+name+"_"+col for col in cross_tab_df.columns]
                cross_tab_df.columns = columns
                cross_tab_df[key] = cross_tab_key_df


                train_df = pd.merge(train_df, cross_tab_df, on=key, how='left')
                test_df = pd.merge(test_df, cross_tab_df, on=key, how='left')

        return train_df, test_df


#このMixinはリークを起こす可能性があるため多発的に使うのは控える．
class MasicFeatureMixin():


    def create_target_encoding(self,train_df,test_df,key,target):
        '''
        :param train_df:
        :param test_df:
        :param key: カテゴリカルのデータ
        :param target: 予測する対象
        :return: train_df,test_df
        '''
        USE_FEATURE = [key,f"TargetEncoding_{key}_std"]
        target_std = train_df.groupby([key])[target].std().reset_index()
        target_std.columns = USE_FEATURE
        train_df = pd.merge(train_df, target_std, on=key, how="left")
        test_df = pd.merge(test_df, target_std, on=key, how="left")

        train_df = train_df[USE_FEATURE]
        test_df = test_df[USE_FEATURE]

        return train_df, test_df