import argparse
import os, sys
import time
import datetime
import json

# Import pytorch dependencies
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
# from torch.optim.lr_scheduler import _LRScheduler
# from tqdm import tqdm_notebook as tqdm
# import matplotlib.pyplot as plt
# from tools.dataloader import CouponDataset, CouponDataset_test
# from model import Net

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
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

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

import xgboost as xgb

params = {'random_forest': {"bootstrap": True, "max_features": 3, "min_samples_leaf": 3, "min_samples_split": 8, "n_estimators": 500, "n_jobs": 8}, 
            'gradient_boosting': {"learning_rate": 0.8, "max_depth": 4, "min_samples_split": 4, "n_estimators": 500}, 
            'adaboost': {"learning_rate": 1.5, "n_estimators": 50}, 
            'mlp': {"activation": "relu", "alpha": 0.05, "batch_size": 64, "early_stopping": True, "hidden_layer_sizes": 250, "learning_rate": "adaptive", "learning_rate_init": 0.05, "max_iter": 300, "n_iter_no_change": 15, "solver": "sgd", "tol": 0.0001, "validation_fraction": 0.1, "verbose": False},
            'xgb': {"n_estimators": 100, "reg_lambda": 0.01}}

params_0 = {'random_forest': {"bootstrap": True, "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 100, "n_jobs": 8},
            'gradient_boosting': {"learning_rate": 0.8, "max_depth": 6, "min_samples_split": 2, "n_estimators": 500},
            'adaboost': {"learning_rate": 1.2, "n_estimators": 50},
            'mlp': {"activation": "relu", "alpha": 0.1, "batch_size": 64, "early_stopping": True, "hidden_layer_sizes": 250, "learning_rate": "adaptive", "learning_rate_init": 0.08, "max_iter": 300, "n_iter_no_change": 15, "solver": "sgd", "tol": 0.0001, "validation_fraction": 0.1, "verbose": False},
            'xgb': {"n_estimators": 500, "reg_lambda": 0.1}
}


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

# def cxvl_train(args):
#     TRAIN_BATCH_SIZE = 64
#     VAL_BATCH_SIZE = 64

#     trainset = CouponDataset(train=True, all=True)
#     # trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=1)

#     valset = CouponDataset(train=False)
#     # valloader = torch.utils.data.DataLoader(trainset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=1)

#     testset = CouponDataset_test()
#     testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     kfold = KFold(n_splits=5, shuffle=True, random_state=234)

#     val_accs = []
#     train_accs = []

#     for fold, (train_ids, test_ids) in enumerate(kfold.split(trainset)):
#         print(f'FOLD {fold}')
#         print('--------------------------------')
        
#         # Sample elements randomly from a given list of ids, no replacement.
#         train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#         test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

#         trainloader = torch.utils.data.DataLoader(
#                       trainset, 
#                       batch_size=TRAIN_BATCH_SIZE, sampler=train_subsampler)
#         valloader = torch.utils.data.DataLoader(
#                         trainset,
#                         batch_size=VAL_BATCH_SIZE, sampler=test_subsampler)

#         net = Net(trainset.data.shape[1])
#         net.apply(init_weights)
#         net = net.to(device)
#         if device =='cuda':
#             print("Train on GPU...")
#         else:
#             print("Train on CPU...")

#         # Initial learning rate
#         INITIAL_LR = 0.08
#         # INITIAL_LR = 0.001
#         # Momentum for optimizer.
#         MOMENTUM = 0.9
#         # Regularization
#         REG = 1e-7
#         # Total number of training epochs
#         EPOCHS = 500
#         # Learning rate decay policy.
#         DECAY_EPOCHS = 2
#         DECAY = 1.00
        
#         CHECKPOINT_PATH = "./saved_model"
#         # FLAG for loading the pretrained model
#         TRAIN_FROM_SCRATCH = True
#         # TRAIN_FROM_SCRATCH = False
#         # Code for loading checkpoint and recover epoch id.
#         CKPT_PATH = "./saved_model/model.h5"

#         print("Training from scratch ...")
#         start_epoch = 0
#         current_learning_rate = INITIAL_LR

#         print("Starting from learning rate %f:" %current_learning_rate)

#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)
#         # optimizer = optim.Adam(net.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=REG)

#         lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

#         global_step = 0
#         best_val_acc = 0

#         for i in range(start_epoch, EPOCHS):
#             # print(datetime.datetime.now())
#             # Switch to train mode
#             net.train()
#             print("Epoch %d:" %i)

#             total_examples = 0
#             correct_examples = 0

#             train_loss = 0
#             train_acc = 0
            
#             # Train the training dataset for 1 epoch.
#             for batch_idx, (inputs, targets) in enumerate(trainloader):
#                 inputs = inputs.to(device)
#                 targets = targets.to(device)
#                 # Your code: Zero the gradient of the optimizer (1 Line)
#                 optimizer.zero_grad()
#                 outputs = net(inputs)
#                 # print(targets, outputs)
#                 # input()
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()
#                 lr_scheduler.step()
#                 # Calculate predicted labels
#                 _, predicted = outputs.max(1)
#                 total_examples += predicted.size(0)
#                 correct_examples += predicted.eq(targets).sum().item()
#                 train_loss += loss
#                 global_step += 1
                        
#             avg_loss = train_loss / (batch_idx + 1)
#             avg_acc = correct_examples / total_examples
#             print("| Training loss:   %.4f, | Training accuracy:   %.4f" %(avg_loss, avg_acc))
#             # print("Validation...")
#             total_examples = 0
#             correct_examples = 0
            
#             net.eval()

#             val_loss = 0
#             val_acc = 0
#             # Disable gradient during validation
#             with torch.no_grad():
#                 for batch_idx, (inputs, targets) in enumerate(valloader):
#                     # Copy inputs to device
#                     inputs = inputs.to(device)
#                     targets = targets.to(device)
#                     # Zero the gradient
#                     optimizer.zero_grad()
#                     # Generate output from the DNN.
#                     outputs = net(inputs)
#                     loss = criterion(outputs, targets)            
#                     # Calculate predicted labels
#                     _, predicted = outputs.max(1)
#                     total_examples += predicted.size(0)
#                     correct_examples += predicted.eq(targets).sum().item()
#                     val_loss += loss

#             avg_loss = val_loss / len(valloader)
#             avg_acc = correct_examples / total_examples
            
#             print("| Validation loss: %.4f, | Validation accuracy: %.4f" % (avg_loss, avg_acc))

#             # Handle the learning rate scheduler.
#             # if i % DECAY_EPOCHS == 0 and i != 0:
#             #     current_learning_rate = current_learning_rate * DECAY
#             #     for param_group in optimizer.param_groups:
#             #         param_group['lr'] = current_learning_rate
#             #     print("Current learning rate has decayed to %f" %current_learning_rate)
            
#             # Save for checkpoint
#             if avg_acc > best_val_acc:
#                 best_val_acc = avg_acc
#                 if not os.path.exists(CHECKPOINT_PATH):
#                     os.makedirs(CHECKPOINT_PATH)
#                 print("Saving ...")
#                 state = {'net': net.state_dict(),
#                         'epoch': i,
#                         'lr': current_learning_rate}
#                 torch.save(state, os.path.join(CHECKPOINT_PATH, 'model.h5'))
#         val_accs.append(best_val_acc)

#     print("Optimization finished.")

#     print('')
#     print('##### Cross validation accuracy = {} #####'.format(np.mean(val_accs)))

def predictor_model_zoo(model_name="random_forest"):
    if model_name == "random_forest":
        predictor = RandomForestClassifier(n_estimators=100,
                                          verbose=False,
                                          n_jobs=8)
    elif model_name == "mlp":
        predictor = MLPClassifier(hidden_layer_sizes=(100),
                                 solver='sgd',
                                 activation='relu',
                                 learning_rate="adaptive",
                                 learning_rate_init=0.08,
                                 max_iter=200,
                                 n_iter_no_change=20,
                                 tol=1e-4,
                                 early_stopping=True,
                                 # validation_fraction=0.10,
                                 batch_size=64,
                                 alpha=0.05,
                                 verbose=True)
    elif model_name == "gpr":
        kernel = DotProduct() + WhiteKernel()
        # predictor = GaussianProcessRegressor(alpha=1e-4, n_restarts_optimizer=10, kernel=RBF(0.1))
        predictor = GaussianProcessClassifier( n_restarts_optimizer=0, kernel=kernel)
    elif model_name == 'adaboost':
        predictor = AdaBoostClassifier(n_estimators=200)
    elif model_name == 'gradient_boosting':
        predictor = GradientBoostingClassifier(n_estimators=80, learning_rate=1.2, max_depth=5, random_state=0)
    elif model_name == "bagging":
        predictor = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)
    elif model_name == "logistic":
        predictor = LogisticRegression(max_iter=200, random_state=0)
    elif model_name == 'gb':
        predictor = GaussianNB()
    elif model_name == 'histgradient':
        predictor = HistGradientBoostingClassifier(max_iter=500, l2_regularization=0.01, max_depth=8, random_state=0)
    elif model_name == "xgb":
        predictor  = xgb.XGBClassifier(n_estimators=150, reg_lambda=0.1)
    elif model_name == "voting":
        clf1 = LogisticRegression(max_iter=200, random_state=0)
        clf2 = GradientBoostingClassifier(n_estimators=80, learning_rate=1.2, max_depth=5, random_state=0)
        clf2.set_params(**params_0['gradient_boosting'])
        clf3 = RandomForestClassifier(n_estimators=100, verbose=False, n_jobs=8)
        clf3.set_params(**(params_0['random_forest']))
        clf4 = AdaBoostClassifier(n_estimators=200)
        clf4.set_params(**params_0['adaboost'])
        clf5 = MLPClassifier(hidden_layer_sizes=(250),
                                 solver='sgd',
                                 activation='relu',
                                 learning_rate="adaptive",
                                 learning_rate_init=0.08,
                                 max_iter=300,
                                 n_iter_no_change=15,
                                 tol=1e-4,
                                 early_stopping=True,
                                 validation_fraction=0.10,
                                 batch_size=64,
                                 alpha=0.01,
                                 verbose=False)
        clf5.set_params(**params_0['mlp'])
        clf6 = xgb.XGBClassifier(n_estimators=150, reg_lambda=0.1)
        clf6.set_params(**params_0['xgb'])
        # predictor = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('mlp', clf4), ('ada', clf5), ('xgb', clf6)], voting='soft')
        # predictor = VotingClassifier(estimators=[('rf', clf2), ('gnb', clf3), ('mlp', clf4), ('ada', clf5), ('xgb', clf6)], voting='soft') # 0.736
        # predictor = VotingClassifier(estimators=[('rf', clf2), ('gnb', clf3), ('mlp', clf4), ('ada', clf5), ('xgb', clf6)], voting='soft')
        predictor = VotingClassifier(estimators=[('rf', clf2), ('gnb', clf3), ('ada', clf5), ('xgb', clf6)], voting='soft')
    else:
        raise NotImplementedError
    # predictor.set_params(**params[model_name])
    return predictor

def param_generate(model_name):
    if model_name == "random_forest":
        param_grid = {
            'bootstrap': [True],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 4, 8],
            'n_estimators': [50, 100, 200, 500],
            'n_jobs' : [8]
        }
        model = RandomForestClassifier()
    elif model_name == 'gradient_boosting':
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 4, 8],
            'n_estimators': [50, 100, 200, 500], 
            'learning_rate': [0.8, 1, 1.2, 1.5],
            'warm_start': [True, False]
        }
        model = GradientBoostingClassifier()
    elif model_name == 'adaboost':
        param_grid = {
            'n_estimators': [50, 100, 200, 500], 
            'learning_rate': [0.8, 1, 1.2, 1.5]
        }
        model = AdaBoostClassifier()
    elif model_name == 'mlp':
        param_grid = {
            'hidden_layer_sizes':[(250), (250, 100), (100)],
            'solver':['sgd'],
            'activation':['relu'],
            'learning_rate':["adaptive"],
            'learning_rate_init':[0.08, 0.05, 0.01],
            'max_iter':[300],
            'n_iter_no_change':[15],
            'tol':[1e-4],
            'early_stopping': [True],
            'validation_fraction': [0.10],
            'batch_size': [64],
            'alpha' : [0.1, 0.05, 0.01, 0.005],
            'verbose': [False]
        }
        model = MLPClassifier()
    elif model_name == "xgb":
        param_grid = {
            'n_estimators': [50, 100, 150, 500], 
            'reg_lambda': [0.1, 0.05, 0.01, 0.005]
        }
        model = xgb.XGBClassifier()
    return model, param_grid

def hyperparameter(args, feature_select):
    predictor = predictor_model_zoo(args['model_type'])

    data_name = '_normal'
    if feature_select:
        data_name = ''
    data = np.load('data/train_data{}.npy'.format(data_name))
    target = np.load('data/train_label{}.npy'.format(data_name))

    print('data/train_data{}.npy'.format(data_name))

    models = ["random_forest", 'gradient_boosting', 'adaboost', 'mlp', 'xgb']

    for model_name in models:
        print('Processing model : ' + model_name)
        model, param_grid = param_generate(model_name)
        clf = GridSearchCV(model, param_grid, n_jobs=8, cv=5)

        clf.fit(data, target)
        print(clf.best_params_)
        print(clf.best_score_)

        with open('param{}.txt'.format(data_name), "a") as f:
            f.write('model is ' + model_name + '\n')
            f.write(json.dumps(clf.best_params_))
            f.write('\n')



def feature_engineering(predictor, data, target):
    print('Processing feature engineering ...')

    svc = AdaBoostClassifier(n_estimators=200)
    # The "accuracy" scoring shows the proportion of correct classifications

    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=svc,
        step=1,
        cv=StratifiedKFold(5),
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
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
    plt.savefig('feature')

    print('End feature engineering ...')

def train_cxvl_predictor(args, feature_select):

    # params = {"bootstrap": True, "max_features": 3, "min_samples_leaf": 3, "min_samples_split": 2, "n_estimators": 500, "n_jobs": 8}

    data_name = '_normal'
    if feature_select:
        data_name = ''

    predictor = predictor_model_zoo(args['model_type'])
    # predictor.set_params(**params_0[args['model_type']])
    data = np.load('data/train_data{}.npy'.format(data_name))
    target = np.load('data/train_label{}.npy'.format(data_name))
    print(data.shape)
    # data = feature_engineering(predictor, data, target)

    val_accs = []

    kfold = KFold(n_splits=5, shuffle=True, random_state=234)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        X_train = data[train_ids]
        y_train = target[train_ids]
        X_test = data[test_ids]
        y_test = target[test_ids]

        print(X_train.shape)
        predictor.fit(X_train, y_train)

        acc = predictor.score(X_test, y_test)
        acc_train = predictor.score(X_train, y_train)
        val_accs.append(acc)

        print('Current Val Acc = {}'.format(acc))
        print('Current Tra Acc = {}'.format(acc_train))
    
    print('')
    print('##### Cross validation accuracy = {} #####'.format(np.mean(val_accs)))    

def train_all_predictor(args, feature_select):
    predictor = predictor_model_zoo(args['model_type'])

    data_name = '_normal'
    if feature_select:
        data_name = ''

    data = np.load('data/train_data{}.npy'.format(data_name))
    target = np.load('data/train_label{}.npy'.format(data_name))


    test_data = np.load('data/test_data{}.npy'.format(data_name))
    scaler = preprocessing.StandardScaler().fit(test_data)
    test_data = scaler.transform(test_data)

    predictor.fit(data, target)
    acc_train = predictor.score(data, target)
    print('Current Tra Acc = {}'.format(acc_train))

    prediction = predictor.predict(test_data)
    with open('results/prediction0.csv', 'w') as f:
        f.write('id,Decision\n')
        for i, item in enumerate(prediction):
            f.write('{},{}\n'.format(i+1, item))
    print(prediction)



# def train(args):

    # trainset = CouponDataset(train=True, all=False)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=1)

    # valset = CouponDataset(train=False)
    # valloader = torch.utils.data.DataLoader(trainset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=1)

    # testset = CouponDataset_test()
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # net = Net()
    # # net.apply(init_weights)
    # net = net.to(device)
    # if device =='cuda':
    #     print("Train on GPU...")
    # else:
    #     print("Train on CPU...")

    # # Initial learning rate
    # INITIAL_LR = 0.001
    # # Momentum for optimizer.
    # MOMENTUM = 0.9
    # # Regularization
    # REG = 1e-4
    # # Total number of training epochs
    # EPOCHS = 100
    # # Learning rate decay policy.
    # DECAY_EPOCHS = 2
    # DECAY = 1.00
    
    # CHECKPOINT_PATH = "./saved_model"
    # # FLAG for loading the pretrained model
    # TRAIN_FROM_SCRATCH = True
    # # TRAIN_FROM_SCRATCH = False
    # # Code for loading checkpoint and recover epoch id.
    # CKPT_PATH = "./saved_model/model.h5"
    # def get_checkpoint(ckpt_path):
    #     try:
    #         ckpt = torch.load(ckpt_path)
    #     except Exception as e:
    #         print(e)
    #         return None
    #     return ckpt

    # ckpt = get_checkpoint(CKPT_PATH)

    # if ckpt is None or TRAIN_FROM_SCRATCH:
    #     if not TRAIN_FROM_SCRATCH:
    #         print("Checkpoint not found.")
    #     print("Training from scratch ...")
    #     start_epoch = 0
    #     current_learning_rate = INITIAL_LR
    # else:
    #     print("Successfully loaded checkpoint: %s" %CKPT_PATH)
    #     net.load_state_dict(ckpt['net'])
    #     start_epoch = ckpt['epoch'] + 1
    #     current_learning_rate = ckpt['lr']
    #     print("Starting from epoch %d " %start_epoch)

    # print("Starting from learning rate %f:" %current_learning_rate)

    # criterion = nn.CrossEntropyLoss()
    # # Your code: use an optimizer
    # # optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)
    # optimizer = optim.Adam(net.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=REG)

    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    # global_step = 0
    # best_val_acc = 0

    # for i in range(start_epoch, EPOCHS):
    #     # print(datetime.datetime.now())
    #     # Switch to train mode
    #     net.train()
    #     print("Epoch %d:" %i)

    #     total_examples = 0
    #     correct_examples = 0

    #     train_loss = 0
    #     train_acc = 0
        
    #     # Train the training dataset for 1 epoch.
    #     for batch_idx, (inputs, targets) in enumerate(trainloader):
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)
    #         # Your code: Zero the gradient of the optimizer (1 Line)
    #         optimizer.zero_grad()
    #         outputs = net(inputs)
    #         # print(targets, outputs)
    #         # input()
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()
    #         # Calculate predicted labels
    #         _, predicted = outputs.max(1)
    #         total_examples += predicted.size(0)
    #         correct_examples += predicted.eq(targets).sum().item()
    #         train_loss += loss
    #         global_step += 1
                    
    #     avg_loss = train_loss / (batch_idx + 1)
    #     avg_acc = correct_examples / total_examples
    #     print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))
    #     print("Validation...")
    #     total_examples = 0
    #     correct_examples = 0
        
    #     net.eval()

    #     val_loss = 0
    #     val_acc = 0
    #     # Disable gradient during validation
    #     with torch.no_grad():
    #         for batch_idx, (inputs, targets) in enumerate(valloader):
    #             # Copy inputs to device
    #             inputs = inputs.to(device)
    #             targets = targets.to(device)
    #             # Zero the gradient
    #             optimizer.zero_grad()
    #             # Generate output from the DNN.
    #             outputs = net(inputs)
    #             loss = criterion(outputs, targets)            
    #             # Calculate predicted labels
    #             _, predicted = outputs.max(1)
    #             total_examples += predicted.size(0)
    #             correct_examples += predicted.eq(targets).sum().item()
    #             val_loss += loss

    #     avg_loss = val_loss / len(valloader)
    #     avg_acc = correct_examples / total_examples
        
    #     print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))

    #     # Handle the learning rate scheduler.
    #     # if i % DECAY_EPOCHS == 0 and i != 0:
    #     #     current_learning_rate = current_learning_rate * DECAY
    #     #     for param_group in optimizer.param_groups:
    #     #         param_group['lr'] = current_learning_rate
    #     #     print("Current learning rate has decayed to %f" %current_learning_rate)
        
    #     # Save for checkpoint
    #     if avg_acc > best_val_acc:
    #         best_val_acc = avg_acc
    #         if not os.path.exists(CHECKPOINT_PATH):
    #             os.makedirs(CHECKPOINT_PATH)
    #         print("Saving ...")
    #         state = {'net': net.state_dict(),
    #                 'epoch': i,
    #                 'lr': current_learning_rate}
    #         torch.save(state, os.path.join(CHECKPOINT_PATH, 'model.h5'))

    # print("Optimization finished.")
    # print("Perform inference on testing set ...")
    # predictions = []
    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(testloader):
    #         # Copy inputs to device
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)
    #         # Zero the gradient
    #         optimizer.zero_grad()
    #         # Generate output from the DNN.
    #         outputs = net(inputs)      
    #         # Calculate predicted labels
    #         _, predicted = outputs.max(1)
    #         for item in predicted.cpu().numpy():
    #             predictions.append(item)
    # predictions = np.asarray(predictions).flatten()
    # print(predictions)
    # print(predictions.shape)
    # with open('results/prediction0.csv', 'w') as f:
    #     f.write('id,Decision\n')
    #     for i, item in enumerate(predictions):
    #         f.write('{},{}\n'.format(i+1, item))