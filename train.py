import argparse
import os, sys
import time
import datetime

# Import pytorch dependencies
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
# from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
# from tools.dataloader import CouponDataset, CouponDataset_test
# from model import Net

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

from sklearn import preprocessing

import xgboost as xgb

# from train_util import train, cxvl_train, train_cxvl_predictor, train_all_predictor, hyperparameter
from train_util import train_cxvl_predictor, train_all_predictor, hyperparameter

def train(args, feature_select):


    if args['whole_training']:
        train_all_predictor(args, feature_select)
    else:
        train_cxvl_predictor(args, feature_select)


def data_preprocess(train=True, data_path="data/test.csv"):
    # 'Decision', 'Driving_to', 'Passanger', 'Weather', 'Temperature', 'Time',
    #    'Coupon', 'Coupon_validity', 'Gender', 'Age', 'Maritalstatus',
    #    'Children', 'Education', 'Occupation', 'Income', 'Direction_same',
    #    'Distance'


    data = pd.read_csv(data_path)
    data.drop(columns=['id'], inplace=True)

    # data.drop(columns=['Driving_to'], inplace=True)
    # data.drop(columns=['Passanger'], inplace=True)
    # data.drop(columns=['Time'], inplace=True)
    # data.drop(columns=['Coupon'], inplace=True)
    # data.drop(columns=['Coupon_validity'], inplace=True)
    # data.drop(columns=['Distance'], inplace=True)

    # data.drop(columns=['Weather'], inplace=True)
    # data.drop(columns=['Temperature'], inplace=True)
    # data.drop(columns=['Age'], inplace=True)
    # data.drop(columns=['Maritalstatus'], inplace=True)
    # data.drop(columns=['Children'], inplace=True)
    
    ##### fix
    # data.drop(columns=['Education'], inplace=True)
    # data.drop(columns=['Occupation'], inplace=True)
    # data.drop(columns=['Income'], inplace=True)
    # data.drop(columns=['Gender'], inplace=True)


    # data.drop(columns=['Direction_same'], inplace=True)

    data = pd.get_dummies(data)
    # print(data.columns[data.isna().any()].tolist())
    # print(data['Bar'].tolist())
    # input()
    data = data.fillna(-1)
    # data = data.dropna(axis='columns')

    return data
    
# def data_preprocess_cxvl():


def data_preprocess_train():
    train_data = data_preprocess(True, 'data/train.csv')

    labels = train_data['Decision']
    # print(labels)
    features = train_data.iloc[:, 1:] 

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    # print(X_train.shape, X_test.shape)
    # input()

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()


    # print(X_train, X_test)
    with open('data/train_data.npy', 'wb') as f:
        np.save(f, X_train)
    with open('data/val_data.npy', 'wb') as f:
        np.save(f, X_test)
    with open('data/train_label.npy', 'wb') as f:
        np.save(f, y_train)
    with open('data/val_label.npy', 'wb') as f:
        np.save(f, y_test)

def data_preprocess_test():
    data_path = 'data/test.csv'
    test_data = data_preprocess(False, data_path)
    X_test = test_data.to_numpy()
    # print(X_test.shape)
    with open('data/test_data.npy', 'wb') as f:
        np.save(f, X_test)

def data_preprocess_all(feature_select=True):
    data_path = 'data/train.csv'
    train_data = data_preprocess(False, data_path)
    features = train_data.iloc[:, 1:] 
    X_train = features.to_numpy()

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    labels = train_data['Decision']
    y_train = labels.to_numpy()

    data_name = '_normal'
    if feature_select:
        data_name = ''
        rfecv, X_train = feature_engineering(X_train, y_train)
    # X_train = feature_engineering1(X_train, y_train)

    with open('data/train_data{}.npy'.format(data_name), 'wb') as f:
        np.save(f, X_train)
    with open('data/train_label{}.npy'.format(data_name), 'wb') as f:
        np.save(f, y_train)

    data_path = 'data/test.csv'
    test_data = data_preprocess(False, data_path)
    X_test = test_data.to_numpy()

    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler.transform(X_test)
    if feature_select:
        X_test = rfecv.transform(X_test)
    print(X_test.shape)
    with open('data/test_data{}.npy'.format(data_name), 'wb') as f:
        np.save(f, X_test)

def feature_engineering(data, target):
    print('Processing feature engineering ...')

    clf2 = GradientBoostingClassifier(n_estimators=80, learning_rate=1.2, max_depth=5, random_state=0)
    svc = RandomForestClassifier(n_estimators=100, verbose=False, n_jobs=8, random_state=0)
    clf4 = AdaBoostClassifier(n_estimators=200)
    # svc = VotingClassifier(estimators=[('rf', clf2), ('gnb', clf3), ('mlp', clf4)], voting='soft')

    # svc = AdaBoostClassifier(n_estimators=200)
    # The "accuracy" scoring shows the proportion of correct classifications

    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=svc,
        step=1,
        cv=StratifiedKFold(5),
        scoring="accuracy",
        n_jobs=8,
        min_features_to_select=min_features_to_select
    )
    rfecv.fit(data, target)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
        rfecv.grid_scores_,
    )
    plt.show()
    plt.savefig('feature_selection', dpi=300)

    print('End feature engineering ...')
    # input()

    X_new = rfecv.transform(data)
    print(X_new.shape)
    return rfecv, X_new

def plot():
    acc = [0.76522, 0.7263, 0.7517, 0.6846, 0.7592, 0.7303]
    model_name = [ 'voting', 'GBT', 'RF', 'Ada', 'XGB', 'MLP']
    plt.bar(model_name,
        acc, 
        width=0.5, 
        bottom=None, 
        align='center', 
        color=['lightsteelblue', 
               'cornflowerblue', 
               'royalblue', 
               'midnightblue', 
               'navy', 
               'darkblue'],
       )
    plt.ylim(top=0.78) #ymax is your value
    plt.ylim(bottom=0.67) #ymin is your value
    plt.title('Cross validation accuracy with different model')
    plt.savefig('acc_model', dpi=300)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=False, default=None,
                          help="test result path.")
    # parser.add_argument("--cxvl", type=int, default=0, help="")
    parser.add_argument("--whole_training", type=int, required=False, default=0,
                        help="")      
    # parser.add_argument("--sklearn", type=int, required=True, default=0)     
    parser.add_argument("--model_type", type=str, required=True, default=None)     
    parser.add_argument("--feature_select", type=int, required=True, default=0)                                

    args = vars(parser.parse_args())
    print(args)
    feature_select = args['feature_select']
    data_preprocess_all(feature_select)
    # data_preprocess_train()
    # data_preprocess_test()
    train(args, feature_select)
    # hyperparameter(args, feature_select)

    # plot()