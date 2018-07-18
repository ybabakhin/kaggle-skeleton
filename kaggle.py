import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import os
from models_zoo import BesXGboost, BesLightGBM, BesCatBoost
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from target_encoding import TargetEncoding
from sklearn.preprocessing import LabelEncoder


class Kaggle:

    def __init__(self, data_path, metric='auc', mode=0):
        self.data_path = data_path
        self.mode = mode
        self.df_train = None
        self.df_test = None
        self.target_colname = None
        self.id_colname = None
        self.sep = None
        self.metric = metric
        self.compute_metric, self.maximize = self.get_metric(metric)

    @staticmethod
    def get_metric(metric):
        if metric == 'auc':
            return metrics.roc_auc_score, True
        if metric == 'mae':
            return metrics.mean_absolute_error, False

    def read_train_data(self, train_name='train.csv', sep=',', target_colname=None, id_colname=None):
        self.sep = sep
        self.df_train = pd.read_csv(os.path.join(self.data_path, train_name), sep=self.sep)
        if target_colname:
            self.target_colname = target_colname
        if id_colname:
            self.id_colname = id_colname

    def read_test_data(self, test_name='test.csv'):
        self.df_test = pd.read_csv(os.path.join(self.data_path, test_name), sep=self.sep)

    def create_validation_split(self, n_folds=5, stratified=False):
        if stratified:
            skf = StratifiedKFold(n_splits=n_folds, random_state=13, shuffle=True)
            idx = 0
            for train_index, test_index in skf.split(self.df_train[[self.id_colname]], self.df_train[[self.target_colname]]):
                self.df_train[[self.id_colname]].loc[train_index, :].to_csv('cv_splits/train_cv_fold_{}'.format(idx), index=False)
                self.df_train[[self.id_colname]].loc[test_index, :].to_csv('cv_splits/test_cv_fold_{}'.format(idx), index=False)
                idx += 1
        else:
            skf = KFold(n_splits=n_folds, random_state=13, shuffle=True)
            idx = 0
            for train_index, test_index in skf.split(self.df_train[[self.id_colname]]):
                self.df_train[[self.id_colname]].loc[train_index, :].to_csv('cv_splits/train_cv_fold_{}'.format(idx), index=False)
                self.df_train[[self.id_colname]].loc[test_index, :].to_csv('cv_splits/test_cv_fold_{}'.format(idx), index=False)
                idx += 1

    def general_feature_engineering(self, train_only=True):
        if train_only:
            df = self.df_train
        else:
            df = pd.concat([self.df_train, self.df_test])
        if self.mode == 0:
            df['category_14_39'] = 0
            for i in range(14, 40):
                df['category_14_39'] = np.where((df[str(i)] == 1) & (df['category_14_39'] == 0), i - 13,
                                                df['category_14_39'])
                df.drop(str(i), 1, inplace=True)

            df['category_128_193'] = 0
            for i in range(128, 194):
                df['category_128_193'] = np.where((df[str(i)] == 1) & (df['category_128_193'] == 0), i - 127,
                                                  df['category_128_193'])
                df.drop(str(i), 1, inplace=True)

            df['category_225_310'] = 0
            for i in range(225, 311):
                df['category_225_310'] = np.where((df[str(i)] == 1) & (df['category_225_310'] == 0), i - 224,
                                                  df['category_225_310'])
                df.drop(str(i), 1, inplace=True)

            # Filling NAs
            # df.smoking_status.fillna('not_answered',inplace=True)
            # Replacing For Binary
            # df.ever_married.replace({'Yes':1,'No':0},inplace=True)
            # Flags
            # df['is_bmi_missed'] = np.where(pd.isnull(df.bmi),1,0)

        elif self.mode == 1:
            pass
        else:
            raise ValueError('There is No Such Feature Engineering Mode')

        if not train_only:
            self.df_train = df.head(self.df_train.shape[0])
            self.df_test = df.tail(self.df_test.shape[0])

    def _categorical_preprocess(self, df, cat_feature, how='ohe'):
        assert how in ['ohe_encoder', 'label_encoder', 'target_encoder']

        if how == 'ohe_encoder':
            df.drop(cat_feature, 1, inplace=True)
            df = df.join(pd.get_dummies(df[cat_feature], prefix=cat_feature))

            # from sklearn.preprocessing import OneHotEncoder
            # ohe = OneHotEncoder(sparse=False)
            # y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))
        elif how == 'label_encoder':
            le = LabelEncoder()
            df[cat_feature] = le.fit_transform(df[cat_feature])
        elif how == 'target_encoder':
            pass
        else:
            raise ValueError('There is no such categorical preprocessing')

        return df

    def fold_feature_engineering(self, train, test):

        df = pd.concat([train, test])

        # Target Encoding
        # te = TargetEncoding(10)
        # train, test = te.fit(train, test, '341', self.target_colname)

        # Delete Duplicated and Constant Columns
        # sys.setrecursionlimit(5500)
        # df = df.loc[:, df.apply(pd.Series.nunique) != 1]
        # df = df.T.drop_duplicates().T

        if self.mode == 0:

            train = df.head(train.shape[0])
            test = df.tail(test.shape[0])

            te = TargetEncoding(10)
            train, test = te.fit(train, test, '341', self.target_colname)
            train, test = te.fit(train, test, 'category_14_39', self.target_colname)
            train, test = te.fit(train, test, 'category_128_193', self.target_colname)
            train, test = te.fit(train, test, 'category_225_310', self.target_colname)

        elif self.mode == 1:
            pass
        else:
            raise ValueError('There is No Such Feature Engineering Mode')

        return train[[col for col in train.columns if col not in [self.target_colname, self.id_colname]]],\
               test[[col for col in test.columns if col not in [self.target_colname, self.id_colname]]]


    def get_predictions(self, model_name, params, X_train, y_train, X_test):

        if model_name == 'xgboost':
            model = BesXGboost(params=params, metric=self.metric, maximize=self.maximize)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            # feat_imp = xgb.feature_importance()
            # print(feat_imp.head())

        elif model_name == 'lightgbm':
            model = BesLightGBM(params=params, metric=self.metric, maximize=self.maximize)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            # feat_imp = lgb.feature_importance()

        elif model_name == 'catboost':
            model = BesCatBoost(params=params, metric=self.metric.upper(), maximize=self.maximize)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        elif model_name == 'logistic_regression':
            model = LogisticRegression()
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_test)[:, 1]

        else:
            raise ValueError('There is No Such Model')

        return model, pred

    def run_single_model_validation(self, model_name='xgboost', params=None, oof_preds_path=''):
        if oof_preds_path != '':
            oof_preds = pd.DataFrame()

        cv_metrics = []
        for fold in range(5):
            print('************************** FOLD {} **************************'.format(fold + 1))
            ids_train = pd.read_csv('cv_splits/train_cv_fold_{}'.format(fold))
            ids_test = pd.read_csv('cv_splits/test_cv_fold_{}'.format(fold))

            df_cv_train, df_cv_test = self.df_train.merge(ids_train), self.df_train.merge(ids_test)
            y_train, y_test = df_cv_train[self.target_colname], df_cv_test[self.target_colname]

            df_cv_train, df_cv_test = self.fold_feature_engineering(df_cv_train, df_cv_test)
            model, pred = self.get_predictions(model_name, params, df_cv_train, y_train, df_cv_test)

            if oof_preds_path != '':
                ids_test[os.path.split(oof_preds_path)[-1].split('.')[0]] = pred
                ids_test[self.target_colname] = y_test
                oof_preds = pd.concat([oof_preds, ids_test])

            met = self.compute_metric(y_test, pred)
            cv_metrics.append(met)
            print(met)

        if oof_preds_path != '':
            oof_preds.to_csv(oof_preds_path, index=False)

        metric_mean = round(np.mean(cv_metrics), 5)
        metric_std = round(np.std(cv_metrics), 5)
        metric_overall = round(np.mean(cv_metrics) - np.std(cv_metrics), 5) if self.maximize else round(np.mean(cv_metrics) + np.std(cv_metrics), 5)
        print('{metric} mean: {mean}, {metric} std: {std}, {metric} overall: {ov}'.format(
            metric=self.metric,
            mean=metric_mean,
            std=metric_std,
            ov=metric_overall))
        print('ALL FOLDS:', [round(x, 5) for x in cv_metrics])
        return metric_mean, metric_std, metric_overall

    def run_stacked_model_validation(self, model_name='logistic_regression', params=None, prev_level_fold='oof_preds_level_1/', oof_preds_path=''):
        if oof_preds_path != '':
            oof_preds = pd.DataFrame()

        cv_metrics = []
        for fold in range(5):
            print('************************** FOLD {} **************************'.format(fold + 1))
            ids_train = pd.read_csv('cv_splits/train_cv_fold_{}'.format(fold))
            ids_test = pd.read_csv('cv_splits/test_cv_fold_{}'.format(fold))

            df_train = pd.DataFrame()
            for f in os.listdir(prev_level_fold):
                path = os.path.join(prev_level_fold, f)
                if df_train.shape[0] == 0:
                    df_train = pd.read_csv(path)
                else:
                    df_train = df_train.merge(pd.read_csv(path))


            df_cv_train, df_cv_test = df_train.merge(ids_train), df_train.merge(ids_test)
            y_train, y_test = df_cv_train[self.target_colname], df_cv_test[self.target_colname]

            df_cv_train = df_cv_train[[col for col in df_cv_train.columns if col not in [self.target_colname, self.id_colname]]]
            df_cv_test = df_cv_test[[col for col in df_cv_test.columns if col not in [self.target_colname, self.id_colname]]]

            model, pred = self.get_predictions(model_name, params, df_cv_train, y_train, df_cv_test)

            if oof_preds_path != '':
                ids_test[os.path.split(oof_preds_path)[-1].split('.')[0]] = pred
                ids_test[self.target_colname] = y_test
                oof_preds = pd.concat([oof_preds, ids_test])

            met = self.compute_metric(y_test, pred)
            cv_metrics.append(met)
            print(met)

        if oof_preds_path != '':
            oof_preds.to_csv(oof_preds_path, index=False)

        metric_mean = round(np.mean(cv_metrics), 5)
        metric_std = round(np.std(cv_metrics), 5)
        metric_overall = round(np.mean(cv_metrics) - np.std(cv_metrics), 5) if self.maximize else round(np.mean(cv_metrics) + np.std(cv_metrics), 5)
        print('{metric} mean: {mean}, {metric} std: {std}, {metric} overall: {ov}'.format(
            metric=self.metric,
            mean=metric_mean,
            std=metric_std,
            ov=metric_overall))
        print('ALL FOLDS:', [round(x, 5) for x in cv_metrics])
        return metric_mean, metric_std, metric_overall

    def find_optimal_params(self, model_name='xgboost'):

        if model_name == 'xgboost':
            opt_params = BesXGboost.find_best_params(self)

        elif model_name == 'lightgbm':
            pass

        elif model_name == 'catboost':
            pass

        elif model_name == 'logistic_regression':
            pass

        else:
            raise ValueError('There is No Such Model')

        return opt_params

    def get_single_model_test_prediction(self, model_name='xgboost', params=None, preds_path=''):
        ids_test = self.df_test[[self.id_colname]] if self.id_colname is self.df_test.columns else self.df_test.reset_index()[['index']]

        y_train = self.df_train[self.target_colname]
        df_train, df_test = self.fold_feature_engineering(self.df_train, self.df_test)

        model, pred = self.get_predictions(model_name, params, df_train, y_train, df_test)

        if preds_path == '':
            raise ValueError('Specify Path for Test Predictions')

        ids_test[os.path.split(preds_path)[-1].split('.')[0]] = pred
        ids_test.to_csv(preds_path, index=False)

        return ids_test

    def get_stacked_model_test_prediction(self, model_name='logistic_regression', params=None,
                                          prev_level_test_fold = 'test_preds_level_1/', preds_path = ''):

        prev_level_train_fold = 'oof' + prev_level_test_fold[4:]
        df_train = pd.DataFrame()
        for f in os.listdir(prev_level_train_fold):
            path = os.path.join(prev_level_train_fold, f)
            if df_train.shape[0] == 0:
                df_train = pd.read_csv(path)
            else:
                df_train = df_train.merge(pd.read_csv(path))

        df_test = pd.DataFrame()
        for f in os.listdir(prev_level_test_fold):
            path = os.path.join(prev_level_test_fold, f)
            if df_test.shape[0] == 0:
                df_test = pd.read_csv(path)
            else:
                df_test = df_test.merge(pd.read_csv(path))

        ids_test = self.df_test[[self.id_colname]] if self.id_colname is self.df_test.columns else self.df_test.reset_index()[['index']]

        y_train = df_train[self.target_colname]

        df_train = df_train[
            [col for col in df_train.columns if col not in [self.target_colname, self.id_colname]]]
        df_test = df_test[
            [col for col in df_test.columns if col not in [self.target_colname, self.id_colname, 'index']]]

        model, pred = self.get_predictions(model_name, params, df_train, y_train, df_test)

        if preds_path == '':
            raise ValueError('Specify Path for Test Predictions')

        ids_test[os.path.split(preds_path)[-1].split('.')[0]] = pred
        ids_test.to_csv(preds_path, index=False)

        return ids_test
