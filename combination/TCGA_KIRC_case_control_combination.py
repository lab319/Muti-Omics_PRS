# TCGA
# KIRC
# case/control
# LightGBM
# combination of new feature as existing prs
# common samples:341
# normal samples:24
# patients:317
# R2:0.9816101511651395

"""
This file is dedicate to improving the predictive accuracy of PRS model based on LightGBM  
by the combination of the existing results in each molecular dataset.
There are two kinds of methods to improve the accuracy.
One is to mix the results of each molecuar dataset.
Another is construct the new model which the new feature is the existing predictive PRS of each molecuar dataset.
"""


# -*- coding:utf-8 -*- 
import time
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore")


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

# input the each PRS csv file
disease = "KIRC"
study = "case_control"
bayes_func = "lgb_cv"
seed = 1
cv = 5
root = "/Users/panjianqiao/Desktop/PRS_data"
# /Users/panjianqiao/Desktop/PRS_data/KIRC_data/stage_control_two/TCGA_KIRC_stage_control_two_methylation_lgb_prs.csv
methylation_prs = pd.read_csv(root + "/" + disease +"_data/" + study + "/TCGA_" + disease +"_" + study + "_methylation_lgb_prs.csv", index_col=0)
mirna_prs = pd.read_csv(root + "/" + disease +"_data/" + study +"/TCGA_" + disease + "_" + study + "_miRNA_lgb_prs.csv", index_col=0)
mrna_prs = pd.read_csv(root + "/" + disease +"_data/" + study +"/TCGA_" + disease + "_" + study + "_mRNA_lgb_prs.csv", index_col=0)
lncrna_prs = pd.read_csv(root + "/" + disease +"_data/" + study +"/TCGA_" + disease + "_" + study + "_lncRNA_lgb_prs.csv", index_col=0)

# check the index of each csv file and select the common samples
methylation_prs_index = list(methylation_prs.index)
mirna_prs_index = list(mirna_prs.index)
mrna_prs_index = list(mrna_prs.index)
lncrna_prs_index = list(lncrna_prs.index)
common_index = set(methylation_prs_index) & set(mirna_prs_index) & set(mrna_prs_index) & set(lncrna_prs_index)
common_index = list(common_index)
print("The number of common samples: ", len(common_index))
normal_samples = 0
patients = 0
for i in range(len(common_index)):
    if common_index[i].split("-")[-1] == "01":  # patient is labelled 01
        patients += 1
    else:  # normal sample is labelled 11
        normal_samples += 1
print("The number of normal samples: ", normal_samples)
print("The number of patients: ", patients)

# combination of new feature as existing prs
prs = pd.concat([methylation_prs, mirna_prs, mrna_prs, lncrna_prs], axis=1, join="inner")
prs = prs.loc[:, ["methy_prs", "mirna_prs", "mrna_prs", "lncrna_prs", "methy_label"]]
prs.rename(columns={"methy_label": "label"}, inplace=True)
print(prs)

# 5-fold cv
skf = KFold(n_splits=cv, random_state=seed)
data, label = prs.iloc[:, :-1], prs.iloc[:, -1]
combination_prs = pd.DataFrame()
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
for train_index, test_index in skf.split(data, label):
    X_train, y_train = data.iloc[train_index, :], label[train_index]
    X_test, y_test = data.iloc[test_index, :], label[test_index]

    # bayes optimization
    # model_bo = BayesianOptimization(model_params[bayes_func]["model"], model_params[bayes_func]["params"],
    #                                 random_state=seed)
    # model_bo.maximize(init_points=10, n_iter=75)
    # params = model_bo.max["params"]
    # model = LGBMRegressor(objective="regression",
    #                       num_leaves=int(params["num_leaves"]),
    #                       n_estimators=100,
    #                       learning_rate=params["learning_rate"],
    #                       max_depth=int(params["max_depth"]),
    #                       max_bin=int(params["max_bin"]),
    #                       min_split_gain=params["min_split_gain"],
    #                       subsample=params["subsample"],
    #                       subsample_freq=int(params["subsample_freq"]),
    #                       colsample_bytree=params["colsample_bytree"],
    #                       min_child_samples=int(params["min_child_samples"]),
    #                       min_child_weight=params["min_child_weight"],
    #                       reg_alpha=params["reg_alpha"],
    #                       reg_lambda=params["reg_lambda"],
    #                       random_state=seed)
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_prs = pd.DataFrame(y_test_pred, index=X_test.index, columns=["combination_prs"])  # the dataframe of combination PRS
    y_test = pd.DataFrame(y_test)
    y_test_df = pd.concat([y_test_prs, y_test], axis=1)
    each_fold_correlation = y_test_df.combination_prs.corr(y_test_df.label)
    print("R2 of each fold：")
    print(each_fold_correlation ** 2)
    combination_prs = combination_prs.append(y_test_prs)
    print("")
label = pd.DataFrame(label)
label_prs_df = pd.concat([label, combination_prs], axis=1)
correlation = label_prs_df.label.corr(label_prs_df.combination_prs)
print(disease + " " + bayes_func + " " + "R2：")
print(correlation ** 2)

print(label_prs_df)

# label_prs_df.to_csv(root + "/" + disease + "_data/combination/model_combination/TCGA_" + disease + "_" + study + "_model_combination.csv")
