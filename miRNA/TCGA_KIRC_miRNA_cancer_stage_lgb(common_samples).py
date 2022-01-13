# KIRC
# stage/control_two
# miRNA
# lightGBM
# R2 : 0.44967646347890616
# -*- coding:utf-8 -*-
import time
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn import linear_model
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVR
from sklearn.utils import shuffle
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore")


def reduce_mem_usage(df, verbose=True):
    """
    # Function to reduce the DF size
    :param df: dataframe
    :param verbose: bool
    :return: dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # memory_usage calculate the bytes of every columns
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def cv_score(model, data, label, cv, random_state):
    """
    cross validation contain random state seed
    :param model: object
    :param data: dataframe
    :param label: series
    :param cv: int
    :param random_state: int
    :return: metric
    """
    metric_li = []
    skf = KFold(n_splits=cv, random_state=random_state)
    for train_index, test_index in skf.split(data, label):
        X_train, y_train = data.iloc[train_index, :], label[train_index]
        X_test, y_test = data.iloc[test_index, :], label[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = -metrics.mean_squared_error(y_test, y_pred)
        metric_li += [mse]
    return np.mean(metric_li)


def lgb_cv(num_leaves, learning_rate, max_depth, max_bin, min_split_gain, subsample, subsample_freq, colsample_bytree, min_child_samples, min_child_weight,
           reg_alpha, reg_lambda):
    """
    object function of bayes optimization
    :param num_leaves:
    :param max_depth:
    :param subsample:
    :param colsample_bytree:
    :param min_child_samples:
    :param reg_alpha:
    :param reg_lambda:
    :return: r2
    """
    val = cv_score(model=LGBMRegressor(objective="regression",
                                       learning_rate=learning_rate,
                                       n_estimators=100,
                                       num_leaves=int(num_leaves),
                                       max_depth=int(max_depth),
                                       max_bin=int(max_bin),
                                       min_split_gain=min_split_gain,
                                       subsample=subsample,
                                       subsample_freq=int(subsample_freq),
                                       colsample_bytree=colsample_bytree,
                                       min_child_samples=int(min_child_samples),
                                       min_child_weight=min_child_weight,
                                       reg_alpha=reg_alpha,
                                       reg_lambda=reg_lambda,
                                       random_state=seed),
                   data=X_train, label=y_train, cv=3, random_state=seed)
    return val


def data_load(disease, omics, type, random_state):
    """
    load and shuffle core dataset
    :param disease:
    :param omics:
    :param seed:
    :return:
    """
    root = "E:\\PJQ\\PRS"
    data_cancer_reader = pd.read_csv(
        root + "\\" + disease + "_data\\" + omics + "\\normal\\" + disease + "_" + omics + "_select_" + type + "_control_two.csv",
        index_col=0, chunksize=5000)
    data_cancer = pd.DataFrame()
    for chunk in data_cancer_reader:
        chunk = reduce_mem_usage(chunk)
        data_cancer = pd.concat([data_cancer, chunk], axis=0)
        del chunk
    data_cancer = pd.DataFrame(data_cancer.values.T, index=data_cancer.columns, columns=data_cancer.index)
    # C:\Users\Lab319 > E:\PJQ\PRS\PRS_data\KIRC_data\TCGA_KIRC_stage_control_two_model_combination.csv
    data_select = pd.read_csv(root + "\\PRS_data\\" + disease + "_data\\combination\\TCGA_" + disease + "_stage_control_two_model_combination.csv", index_col=0)
    data_cancer = data_cancer.loc[data_select.index, :]
    # shuffle the core dataset
    data_cancer = shuffle(data_cancer, random_state=random_state)
    print(data_cancer.info())
    return data_cancer


if __name__ == "__main__":
    seed = 1
    cv = 5
    bayes_func = "lgb_cv"
    disease = "KIRC"
    omic = "miRNA"
    type = "stage"
    model_params = {"lgb_cv": {"model": lgb_cv, "params": {"num_leaves": (2, 50),
                                                           "learning_rate": (0.005, 0.3),
                                                           "max_depth": (2, 8),
                                                           "max_bin": (80, 500),
                                                           "min_split_gain": (0, 1),
                                                           "subsample": (0.5, 1),
                                                           "subsample_freq": (1, 7),
                                                           "colsample_bytree": (0.1, 1),
                                                           "min_child_samples": (5, 100),
                                                           "min_child_weight": (0.001, 1),
                                                           "reg_alpha": (0, 100),
                                                           "reg_lambda": (0, 100)}}}
    data_cancer = data_load(disease, omic, type, seed)
    skf = KFold(n_splits=cv, random_state=seed)
    prs = pd.DataFrame()
    data, label = data_cancer.iloc[:, :-1], data_cancer.iloc[:, -1]
    start_time = time.time()
    for train_index, test_index in skf.split(data, label):
        X_train, y_train = data.iloc[train_index, :], label[train_index]
        X_test, y_test = data.iloc[test_index, :], label[test_index]
        # bayes optimization
        # model_bo = BayesianOptimization(model_params[bayes_func]["model"], model_params[bayes_func]["params"], random_state=seed)
        # model_bo.maximize(init_points=10, n_iter=75)
        # params = model_bo.max["params"]
        model = LGBMRegressor(objective="regression",
                              num_leaves=31,
                              n_estimators=100,
                              learning_rate=0.1,
                              max_depth=-1,
                              colsample_bytree=1.0,
                              min_split_gain=0,
                              min_child_samples=20,
                              min_child_weight=1e-3,
                              reg_alpha=0,
                              reg_lambda=0)
        # model = LGBMRegressor()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_test_prs = pd.DataFrame(y_test_pred, index=X_test.index, columns=["prs"])
        y_test = pd.DataFrame(y_test)
        y_test_df = pd.concat([y_test_prs, y_test], axis=1)
        each_fold_correlation = y_test_df.prs.corr(y_test_df.label)
        print("R2 of each fold：")
        print(each_fold_correlation ** 2)
        prs = prs.append(y_test_prs)
        print("")

    label = pd.DataFrame(label)
    label_prs_df = pd.concat([label, prs], axis=1)

    correlation = label_prs_df.label.corr(label_prs_df.prs)
    print(disease + " " + omic + " " + "R2：")
    print(correlation ** 2)
    end_time = time.time()
    print("Cost time:", (end_time - start_time)/60, "minutes")

    label_prs_df.rename(columns={"prs": "methy_prs", "label": "methy_label"}, inplace=True)
    # label_prs_df.to_csv("E:\\PJQ\\PRS\\PRS_data\\" + disease + "_data\\" + "TCGA_" + disease + "_" + type + "_control_two_" + omic + "_lgb_prs.csv")

