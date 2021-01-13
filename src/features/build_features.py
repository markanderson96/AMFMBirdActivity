#!/bin/python

import soundfile as sf
import numpy as np
import pandas as pd
import yaml
import logging
import os

from scipy import signal
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import amplitudemod as am
import frequencymod as fm

def main():
    """Creates features from data processed and stored in processed"""
    dataset = config['general']["dataset"]
    data_dir = 'data/processed/'
    label_loc = 'data/raw/' + dataset + '/metadata/labels.csv'
    feature_dir = 'data/features'

    labels = pd.read_csv(label_loc, index_col=None)

    df = pd.DataFrame(columns=[
        'fileIndex', 'amDetected', 'amFreq', 'amDepth', 'amProminence',
        'pitchMean', 'pitchVar', 'pitchSkew', 'pitchKurtosis', 'hasBird'
    ])

    # extract features for each file
    for file in os.listdir(data_dir + dataset):
        # load file
        data, samplerate = sf.read(data_dir + dataset + file)
        logging.info('Processing Features: ' + file)
        
        # AM
        am_min_mod = config['AM']['min_mod']
        am_max_mod = config['AM']['max_mod']
        am_prominence_cutoff = config['AM']['prominence_cutoff']
        am = am.AM(data=data, samplerate=samplerate, min_mod=am_min_mod, 
                max_mod=am_max_mod, prominence_cutoff=am_prominence_cutoff)
        [amDetected, amFreq, amDepth, amProminence] = am.calcAM()

        
        sf.write(data_dir + dataset + '/wav/' + file +'_env.wav', 
                 am.getEnv(), 
                 am.envelope_Fs)

        # FM
        fm_window_length = config['FM']['window_length']
        fm_window_overlap = config['FM']['window_overlap']
        fm_threshold = config['FM']['threshold']
        fm = fm.FM(data=data, 
                   samplerate=samplerate,
                   window_length=fm_window_length, 
                   window_overlap=fm_window_overlap, 
                   threshold=fm_threshold)
        [pitchMeans, pitchVar, pitchSkew, pitchKurtosis] = fm.calcFM()

        features = {
            'amDetected': amDetected,
            'amFreq': amFreq,
            'amDepth': amDepth,
            'amProminence': amProminence,
            'pitchMean': pitchMeans,
            'pitchVar': pitchVar,
            'pitchSkew': pitchSkew,
            'pitchKurtosis': pitchKurtosis,
        }

        features = pd.DataFrame(features)
        df = df.append(features, ignore_index=True)

    # add file indexes and label to feature data frame
    df["fileIndex"] = labels["fileIndex"]
    df["hasBird"] = labels["hasBird"]

    df.to_csv(feature_dir + '/' + dataset + '_features.csv')

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