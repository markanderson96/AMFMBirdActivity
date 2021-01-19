#!/bin/python

import pandas as pd
import numpy as np
import xgboost as xgb
import yaml
import logging

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from sklearn.externals import joblib
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
    features = 'data/features/' + dataset + '_features.csv'

    df = pd.read_csv(features, index_col=None)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df.pitchMean != 0]
    

    categoric_features = ['amDetected']
    numeric_features = [
        'amFreq',
        'amDepth',
        'amProminence',
        'pitchMean',
        'pitchVar',
        'pitchSkew',
        'pitchKurtosis',
    ]

    categoric_data = df[categoric_features]
    numeric_data = df[numeric_features]
    
    #basic scaling
    #numeric_data = numeric_data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    # put data back together
    x_data = categoric_data.join(numeric_data)
    print(x_data)

    # labels
    y_data = df['hasBird']

    # Split data 80/20
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=config['classifier']['test_size'],
                                                        random_state=config['classifier']['random_state'],
                                                        shuffle=config['classifier']['shuffle'])

    classifier_type = config['classifier']['type']

    if classifier_type == 'rf':
        # Create RF classifier
        if config['classifier']['grid_search']:
            parameters = {
                "criterion":["gini", "entropy"],
                "n_estimators":[10, 20, 50, 100, 200, 500],
                "max_depth":[2, 4, 6, 8, 16, 32, None],
                "min_samples_leaf":[1, 2, 4, 8, 16, 32, 64, 128]
            }
            rfc = RandomForestClassifier(n_jobs=-1)
            clf = GridSearchCV(estimator=rfc, 
                            param_grid=parameters, 
                            scoring=config['classifier']['scoring'], 
                            cv=5)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            print(clf.best_score_)
            print(clf.best_params_)
            quit()
        else:
            clf = RandomForestClassifier(verbose=config['classifier']['verbose'],
                                        criterion=config['random_forest']['criterion'],
                                        max_depth=config['random_forest']['max_depth'],
                                        min_samples_leaf=config['random_forest']['min_samples_leaf'],
                                        n_estimators=config['random_forest']['n_estimators'],
                                        n_jobs=-1)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            report(y_test, y_pred)

    elif classifier_type == 'svm':
        # Create a SVM Classifier
        if config['classifier']['grid_search']:
            parameters = {
                "kernel":["linear", "poly", "rbf", 'sigmoid'],
                "degree":[2, 3, 4],
                "gamma":['auto', 'scale'],
            }
            svm = SVC()
            clf = GridSearchCV(estimator=svm, 
                            param_grid=parameters, 
                            scoring=config['classifier']['scoring'], 
                            cv=5)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            print(clf.best_score_)
            print(clf.best_params_)
            quit()
        else:
            clf = SVC(verbose=config['classifier']['verbose'],
                      kernel=config['svm']['kernel'],
                      degree=config['svm']['degree'],
                      gamma=config['svm']['gamma'],
                      probability=config['svm']['probability'],
                      max_iter=config['svm']['max_iter'])
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            report(y_test, y_pred)

    elif classifier_type == 'xgb':
        # Create a xgb Classifier
        if config['classifier']['grid_search']:
            parameters = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5],
                'subsample': [0.1, 0.2, 0.5, 1],
                'colsample_bytree': [0.8, 1.0, 1.2, 1.4],
                'max_depth': [5, 6, 7],
                'learning_rate': [0.01, 0.02, 0.005],
            }
            xgbc = xgb.XGBClassifier(objective='binary:logistic',
                                    use_label_encoder=False,
                                    eval_metric='logloss',
                                    n_jobs=-1)
            clf = GridSearchCV(estimator=xgbc, 
                            param_grid=parameters, 
                            scoring=config['classifier']['scoring'], 
                            cv=5)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            print(clf.best_score_)
            print(clf.best_params_)
            quit()
        else:
            xgbc = xgb.XGBClassifier(verbosity=config['classifier']['verbose'],
                                    booster=config['xgb']['booster'],
                                    gamma=config['xgb']['gamma'],
                                    colsample_bytree=config['xgb']['colsample_bytree'],
                                    subsample=config['xgb']['subsample'],
                                    min_child_weight=config['xgb']['min_child_weight'],
                                    learning_rate=config['xgb']['lr'],
                                    max_depth=config['xgb']['max_depth'],
                                    n_estimators=config['xgb']['n_estimators'],
                                    eval_metric=config['xgb']['eval_metric'],
                                    scale_pos_weight=config['xgb']['scale_pos_weight'],
                                    objective='binary:logistic',
                                    use_label_encoder=False,
                                    n_jobs=-1)
            
            xgbc.fit(x_train, y_train)
            
            scores = cross_val_score(xgbc, x_train, y_train, cv=5)
            print("Mean cross-validation score: %.2f" % scores.mean())
            kfold = KFold(n_splits=10, shuffle=True)
            kf_cv_scores = cross_val_score(xgbc, x_train, y_train, cv=kfold )
            print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
            
            xgb.plot_importance(xgbc)
            pyplot.show()

            y_pred = xgbc.predict(x_test)

            report(y_test, y_pred)

    else:
        raise Exception('Please pick a valid classifier type [rf/svm/xgb]')


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
    # ROC_AUC
    fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
    print("AUC:", metrics.auc(fpr, tpr))

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load config file
    with open(str(project_dir) + '/config/config.yaml') as file:
        config = yaml.safe_load(file)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()