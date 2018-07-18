import pandas as pd
import numpy as np
import operator
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, cv, Pool
import seaborn as sns
import matplotlib.pyplot as plt
import math

color = sns.color_palette()


class BesXGboost:
    """
    XGBoost model. https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    params = {
      'silent': 1 if self.silent else 0,
      'use_buffer': int(self.use_buffer),
      'num_round': self.num_round,
      'ntree_limit': self.ntree_limit,
      'nthread': self.nthread,
      'booster': self.booster,
      'eta': self.eta,
      'gamma': self.gamma,
      'max_depth': self.max_depth,
      'min_child_weight': self.min_child_weight,
      'subsample': self.subsample,
      'colsample_bytree': self.colsample_bytree,
      'max_delta_step': self.max_delta_step,
      'l': self.l,
      'alpha': self.alpha,
      'lambda_bias': self.lambda_bias,
      'objective': self.objective,
      'eval_metric': self.eval_metric,
      'seed': self.seed,
      'num_class': self.num_class,
    }

    xgb_params = {
            'booster': 'gbtree',
            'eta': .1,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'seed': 123,
            'nthread': 3,
            'max_depth': 6,
            'min_child_weight': .1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'silent': 1
        }

    """


    def __init__(self, params, metric='auc', maximize=True, verbose=True, features=None, model=None):
        assert params['booster'] in ['gbtree', 'gblinear']
        assert params['objective'] in ['reg:linear', 'reg:logistic',
                             'binary:logistic', 'binary:logitraw', 'multi:softmax',
                             'multi:softprob', 'rank:pairwise']
        assert params['eval_metric'] in [None, 'rmse', 'mlogloss', 'logloss', 'error',
                               'merror', 'auc', 'ndcg', 'map', 'ndcg@n', 'map@n']

        self.params = params
        self.metric = metric
        self.maximize = maximize
        self.verbose = verbose
        self.features = features
        self.model = model

    def fit(self, X_train, y_train):
        self.features = X_train.columns
        dtrain = xgb.DMatrix(data=X_train, label=y_train)

        if self.verbose:
            bst = xgb.cv(self.params, dtrain, num_boost_round=10000, nfold=3, early_stopping_rounds=50, verbose_eval=50)
        else:
            bst = xgb.cv(self.params, dtrain, num_boost_round=10000, nfold=3, early_stopping_rounds=50)

        if self.maximize:
            best_rounds = int(np.argmax(bst['test-' + self.metric + '-mean'] - bst['test-' + self.metric + '-std']) * 1.5)
        else:
            best_rounds = int(np.argmin(bst['test-' + self.metric + '-mean'] + bst['test-' + self.metric + '-std']) * 1.5)

        if self.verbose:
            print('Best Iteration: {}'.format(best_rounds))

        self.model = xgb.train(self.params, dtrain, best_rounds)

    def predict(self, X_test):
        dtest = xgb.DMatrix(data=X_test)
        pred_prob = self.model.predict(dtest)
        return pred_prob

    def feature_importance(self):
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in self.features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()
        importance = self.model.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        imp = pd.DataFrame(importance, columns=['feature', 'fscore'])
        imp = imp.sort_values(['fscore'], ascending=False)

        # import xgbfir
        # xgbfir.saveXgbFI(model, feature_names=X_train.columns, OutputXlsxFile='irisFI.xlsx',TopK=300)

        return imp

    def _optimize_single_param(self):
        pass

    @staticmethod
    def find_best_params(kag):
        """
        lambda [default=1]

            L2 regularization term on weights (analogous to Ridge regression)
            This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.

        alpha [default=0]

            L1 regularization term on weight (analogous to Lasso regression)
            Can be used in case of very high dimensionality so that the algorithm runs faster when implemented

        scale_pos_weight [default=1]

            A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.
        """
        nthread = 3
        lr = 0.3
        bst = -math.inf if kag.maximize else math.inf
        seed = 123

        params = {
            'booster': 'gbtree',
            'eta': lr,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'seed': seed,
            'nthread': nthread,
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': kag.metric,
            'lambda': 1,
            'alpha': 0,
            'gamma': 0
        }

        for depth in [2, 4, 6, 8, 10]:
            for mcw in [1, 3, 5, 7, 9]:
                print(depth, mcw)
                params['max_depth'] = depth
                params['min_child_weight'] = mcw

                met = kag.run_single_model_validation(model_name='xgboost', params=params)[0]
                cond = met > bst if kag.maximize else met < bst

                if cond:
                    bst = met
                    depth_bst = depth
                    mcw_bst = mcw
        print('Best Depth: {}. Best MCW: {}'.format(depth_bst, mcw_bst))
        print('Score:', bst)

        depth_bst_prev = depth_bst
        mcw_bst_prev = mcw_bst
        for depth in [depth_bst_prev - 1, depth_bst_prev, depth_bst_prev + 1]:
            for mcw in [mcw_bst_prev - 1, mcw_bst_prev, mcw_bst_prev + 1]:
                print(depth, mcw)
                params['max_depth'] = depth
                params['min_child_weight'] = mcw

                met = kag.run_single_model_validation(model_name='xgboost', params=params)[0]
                cond = met > bst if kag.maximize else met < bst

                if cond:
                    bst = met
                    depth_bst = depth
                    mcw_bst = mcw
        print('Best Depth: {}. Best MCW: {}'.format(depth_bst, mcw_bst))
        print('Score:', bst)
        params['max_depth'] = depth_bst
        params['min_child_weight'] = mcw_bst

        colsample_bytree_bst = 0.8
        subsample_bst = 0.8
        for colsample_bytree in [0.4, 0.6, 0.8]:
            for subsample in [0.4, 0.6, 0.8]:
                print(colsample_bytree, subsample)
                params['colsample_bytree'] = colsample_bytree
                params['subsample'] = subsample

                met = kag.run_single_model_validation(model_name='xgboost', params=params)[0]
                cond = met > bst if kag.maximize else met < bst

                if cond:
                    bst = met
                    colsample_bytree_bst = colsample_bytree
                    subsample_bst = subsample
        print('Best Colsample: {}. Best Subsample: {}'.format(colsample_bytree_bst, subsample_bst))
        print('Score:', bst)

        colsample_bytree_bst_prev = colsample_bytree_bst
        subsample_bst_prev = subsample_bst
        for colsample_bytree in [colsample_bytree_bst_prev - 0.1, colsample_bytree_bst_prev,
                                 colsample_bytree_bst_prev + 0.1]:
            for subsample in [subsample_bst_prev - 0.1, subsample_bst_prev, subsample_bst_prev + 0.1]:
                print(colsample_bytree, subsample)
                params['colsample_bytree'] = colsample_bytree
                params['subsample'] = subsample

                met = kag.run_single_model_validation(model_name='xgboost', params=params)[0]
                cond = met > bst if kag.maximize else met < bst

                if cond:
                    bst = met
                    colsample_bytree_bst = colsample_bytree
                    subsample_bst = subsample
        print('Best Colsample: {}. Best Subsample: {}'.format(colsample_bytree_bst, subsample_bst))
        print('Score:', bst)
        params['colsample_bytree'] = colsample_bytree_bst
        params['subsample'] = subsample_bst

        # alpha_bst = 0
        # lamb_bst = 1
        # for alpha in [0, 0.1, 0.5, 1]:
        #     for lamb in [0, 0.1, 0.5, 1]:
        #         print(alpha, lamb)
        #         params['alpha'] = alpha
        #         params['lambda'] = lamb
        #
        #         met = kag.run_single_model_validation(model_name='xgboost', params=params)[2]
        #         cond = met > bst if kag.maximize else met < bst
        #
        #         if cond:
        #             bst = met
        #             alpha_bst = alpha
        #             lamb_bst = lamb
        # print('Best Alpha: {}. Best Lambda: {}'.format(alpha_bst, lamb_bst))
        # print('Score:', bst)
        # params['alpha'] = alpha_bst
        # params['lambda'] = lamb_bst

        lamb_bst = 1
        for lamb in [0, 0.1, 0.5, 1, 5, 10]:
            print(lamb)
            params['lambda'] = lamb

            met = kag.run_single_model_validation(model_name='xgboost', params=params)[0]
            cond = met > bst if kag.maximize else met < bst

            if cond:
                bst = met
                lamb_bst = lamb
        print('Best Lambda: {}'.format(lamb_bst))
        print('Score:', bst)
        params['lambda'] = lamb_bst

        gamma_bst = 0
        for gamma in [0, 0.1, 0.5, 1, 10]:
            print(gamma)
            params['gamma'] = gamma

            met = kag.run_single_model_validation(model_name='xgboost', params=params)[0]
            cond = met > bst if kag.maximize else met < bst

            if cond:
                bst = met
                gamma_bst = gamma
        print('Best Gamma: {}'.format(gamma_bst))
        print('Score:', bst)
        params['gamma'] = gamma_bst

        print(params)
        print('Score:', bst)
        return params


class BesLightGBM:
    """
    lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

    """

    def __init__(self, params, metric='auc', maximize=True, verbose=True, model=None):
        self.params = params
        self.metric = metric
        self.maximize = maximize
        self.verbose = verbose
        self.model = model

    def fit(self, X_train, y_train):
        dtrain = lgb.Dataset(data=X_train, label=y_train)

        if self.verbose:
            bst = lgb.cv(self.params, dtrain, num_boost_round=10000, nfold=3, early_stopping_rounds=50, verbose_eval=50)
        else:
            bst = lgb.cv(self.params, dtrain, num_boost_round=10000, nfold=3, early_stopping_rounds=50)

        if self.maximize:
            best_rounds = int(np.argmax(np.array(bst[self.metric + '-mean']) - np.array(bst[self.metric + '-stdv'])) * 1.5)
        else:
            best_rounds = int(np.argmin(np.array(bst[self.metric + '-mean']) + np.array(bst[self.metric + '-stdv'])) * 1.5)

        if self.verbose:
            print('Best Iteration: {}'.format(best_rounds))

        self.model = lgb.train(self.params, dtrain, best_rounds)

    def predict(self, X_test):
        pred_prob = self.model.predict(X_test)
        return pred_prob

    def feature_importance(self):
        lgb.plot_importance(self.model, max_num_features=10)
        plt.show()
        return self.model.feature_importance()

    @staticmethod
    def find_best_params(kag):
        pass


class BesCatBoost:
    """
    catboost_params = {
            'iterations': 500,
            'depth': 3,
            'learning_rate': 0.1,
            'eval_metric': 'AUC',
            'random_seed': 42,
            'logging_level': 'Verbose',
            'l2_leaf_reg': 15.0,
            'bagging_temperature': 0.75,
            'allow_writing_files': False,
            'metric_period': 50
        }
        """

    def __init__(self, params, metric='AUC', maximize=True, verbose=True, model=None):
        self.params = params
        self.metric = metric
        self.maximize = maximize
        self.verbose = verbose
        self.model = model

    def fit(self, X_train, y_train):

        bst = cv(
            Pool(X_train, y_train),
            self.params
        )

        best_rounds = int(bst['test-{}-mean'.format(self.metric)].idxmax() * 1.5) + 1
        print('Best Iteration: {}'.format(best_rounds))

        self.params['iterations'] = best_rounds
        self.model = CatBoostClassifier(**self.params)

        self.model.fit(
            X_train, y_train
        )

    def predict(self, X_test):
        pred_prob = self.model.predict_proba(X_test)[:, -1]
        return pred_prob

    def feature_importance(self):
        pass

    @staticmethod
    def find_best_params(kag):
        pass

