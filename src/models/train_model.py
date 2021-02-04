#!/bin/python

import pandas as pd
import numpy as np
import yaml
import logging
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn import metrics

from matplotlib import pyplot
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def main():
    """ Trains model based on parameters in config
    """
    logger = logging.getLogger(__name__)
    logger.info('training model')

    dataset = config['general']['dataset']
    features = 'data/features/' + dataset + '/features.csv'

    df = pd.read_csv(features, index_col=None)
    numeric_features = [
        'amFreq',
        'amDepth',
        'amProminence',
        'pitchMean',
        'pitchVar',
        'pitchSkew',
        'pitchKurtosis',
        'spectralCentroidMean',
        'spectralCentroidVar',
        'spectralRolloffMean',
        'spectralRolloffVar'
    ]

    numeric_data = df[numeric_features]
    
    # #basic scaling
    # numerical_transformer = Pipeline(steps=[
    #     ('Imputer', SimpleImputer(strategy='median', verbose=1)),
    #     ('Scaler', StandardScaler())], 
    #     verbose=False)

    # # Preprocessor operations
    # preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('Numerical Data', numerical_transformer, numeric_features)
    #         ],
    #         verbose=False)

    # clf1 = Pipeline(steps=[
    #     ('Preprocessor',    preprocessor),
    #     ('Random Forest',   RandomForestClassifier(n_jobs=-1))],
    #     verbose=False)
    # clf2 = Pipeline(steps=[
    #     ('Preprocessor',    preprocessor),
    #     ('SVM',             SVC(verbose=config['classifier']['verbose'],
    #                             kernel=config['svm']['kernel'],
    #                             degree=config['svm']['degree'],
    #                             gamma=config['svm']['gamma'],
    #                             probability=config['svm']['probability'],
    #                             max_iter=config['svm']['max_iter']))],
    #     verbose=False)

    # put data back together
    x_data = numeric_data

    # labels
    y_data = df['hasBird']

    # Split data 80/20
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=config['classifier']['test_size'],
                                                        random_state=config['classifier']['random_state'],
                                                        shuffle=config['classifier']['shuffle'])


    # eclf = StackingClassifier(estimators=[
    #                             ('rf', clf1), ('svm', clf2)],
    #                             final_estimator=SVC(),
    #                             n_jobs=-1,
    #                             passthrough=False,
    #                             verbose=0)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1100, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 8, num = 6)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4, 8, 16]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    params_rf = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    grid_rf = GridSearchCV(estimator=RandomForestClassifier(), cv=5, n_jobs=-1, param_grid=params_rf, scoring='f1')

    grid_rf.fit(x_train, y_train)
    print(grid_rf.best_params_)

    # Number of trees in random forest
    c = [x for x in np.linspace(start = 0.01, stop = 10, num = 10)]
    # Number of features to consider at every split
    kernel = ['linear', 'poly', 'rbf']
    # Maximum number of levels in tree
    gamma = ['scale', 'auto']
    # Minimum number of samples required to split a node
    order = [2, 3, 4, 5]
    # Create the random grid
    params_svm = {'c': c,
                'kernel': kernel,
                'gamma': gamma,
                'order': order}

    grid_svm = GridSearchCV(estimator=RandomForestClassifier(), cv=5, n_jobs=-1, param_grid=params_svm, scoring='f1')

    grid_svm.fit(x_train, y_train)
    print(grid_svm.best_params_)

    #y_pred_rf = clf1.predict(x_test)
    #print('Random Forest')
    #report(y_test, y_pred_rf)

    # clf2.fit(x_train, y_train)
    # y_pred_svm = clf2.predict(x_test)
    # print('SVM')
    # report(y_test, y_pred_svm)

    # eclf.fit(x_train, y_train)
    # y_pred_combo = eclf.predict(x_test)
    # print('Combo')
    # report(y_test, y_pred_combo)

    # with open('results.csv', 'a') as csv_file:
    #     csv_file.write('{},{},{}\n'.format(metrics.f1_score(y_test, y_pred_rf),
    #                                        metrics.f1_score(y_test, y_pred_svm),
    #                                        metrics.f1_score(y_test, y_pred_combo)))
    # csv_file.close()

def report(y_test, y_pred):
    # Reports
    print()
    print(metrics.classification_report(y_test, y_pred, target_names=['noBird', 'bird']))
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))
    # F1 Score
    print("F1 Score:", metrics.f1_score(y_test, y_pred))
    # ROC_AUC
    fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
    print("AUC:", metrics.auc(fpr, tpr))

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stderr)

    # load config file
    with open(str(project_dir) + '/config/config.yaml') as file:
        config = yaml.safe_load(file)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()