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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
    
    #basic scaling
    numerical_transformer = Pipeline(steps=[
        ('Imputer', SimpleImputer(strategy='median', verbose=1)),
        ('Scaler', StandardScaler())], 
        verbose=False)

    # Preprocessor operations
    preprocessor = ColumnTransformer(
            transformers=[
                ('Numerical Data', numerical_transformer, numeric_features)
            ],
            verbose=False)

    clf1 = Pipeline(steps=[
        ('Preprocessor',    preprocessor),
        ('Random Forest',   RandomForestClassifier(verbose=config['classifier']['verbose'],
                                criterion=config['random_forest']['criterion'],
                                max_depth=config['random_forest']['max_depth'],
                                min_samples_leaf=config['random_forest']['min_samples_leaf'],
                                n_estimators=config['random_forest']['n_estimators'],
                                class_weight='balanced',
                                n_jobs=-1))],
        verbose=False)
    clf2 = Pipeline(steps=[
        ('Preprocessor',    preprocessor),
        ('SVM',             SVC(verbose=config['classifier']['verbose'],
                                kernel=config['svm']['kernel'],
                                degree=config['svm']['degree'],
                                gamma=config['svm']['gamma'],
                                probability=config['svm']['probability'],
                                max_iter=config['svm']['max_iter']))],
        verbose=False)

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


    eclf = StackingClassifier(estimators=[
                                ('rf', clf1), ('svm', clf2)],
                                final_estimator=SVC(),
                                n_jobs=-1,
                                passthrough=False,
                                verbose=0)

    clf1.fit(x_train, y_train)
    y_pred = clf1.predict(x_test)
    print('Random Forest')
    report(y_test, y_pred)

def report(y_test, y_pred):
    # Reports
    print()
    print(metrics.classification_report(y_test, y_pred, target_names=['noBird', 'bird']))
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model F1
    print("F1-Score:", metrics.f1_score(y_test, y_pred))
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
    logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stderr)

    # load config file
    with open(str(project_dir) + '/config/config.yaml') as file:
        config = yaml.safe_load(file)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()