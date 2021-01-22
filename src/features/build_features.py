#!/bin/python

import soundfile as sf
import numpy as np
import pandas as pd
import yaml
import logging
import glob
import os
import multiprocessing

from scipy import signal
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from amplitudemod import AM
from frequencymod import FM
from parametric import parametric

def runProcess(file_queue):
    while not file_queue.empty():
        filename = file_queue.get(block=False)
        makeFeatures(filename)

def makeFeatures(file):
    # load file
    data, samplerate = sf.read(data_dir + os.sep + file)
    logging.info('Processing Features: ' + file)
    

    # add file indexes and label to feature data frame
    labels = pd.read_csv(label_loc, index_col=None)
    labels = labels[labels['fileIndex'].str.contains(file[:-4])]
    labels = labels.reset_index(drop=True)
    if (labels.empty):
        exit
    
    # AM
    am = AM(data=data, samplerate=samplerate, min_mod=am_min_mod, 
            max_mod=am_max_mod, prominence_cutoff=am_prominence_cutoff)
    [amDetected, amFreq, amDepth, amProminence] = am.calcAM()

    sf.write(env_dir + file[:-4] +'_env.wav', 
                am.getEnv(), 
                am.envelope_Fs)

    # FM
    fm = FM(data=data, 
                samplerate=samplerate,
                window_length=fm_window_length, 
                window_overlap=fm_window_overlap, 
                threshold=fm_threshold)
    [pitchMeans, pitchVar, pitchSkew, pitchKurtosis] = fm.calcFM()

    # parametric features
    pm = parametric(data=data, samplerate=samplerate)
    [spectralCentroid, spectralRolloff] = pm.parametricFeatures()

    features = {
        'amDetected': amDetected,
        'amFreq': amFreq,
        'amDepth': amDepth,
        'amProminence': amProminence,
        'pitchMean': pitchMeans,
        'pitchVar': pitchVar,
        'pitchSkew': pitchSkew,
        'pitchKurtosis': pitchKurtosis,
        'spectralCentroid': spectralCentroid,
        'spectralRolloff': spectralRolloff
    }

        # create feature dataframe
    df = pd.DataFrame(features, columns=[
        'fileIndex', 
        'hasBird',
        'amDetected',
        'amFreq',
        'amDepth',
        'amProminence',
        'pitchMean',
        'pitchVar',
        'pitchSkew',
        'pitchKurtosis',
        'spectralCentroid',
        'spectralRolloff'
    ])
    df["fileIndex"] = labels["fileIndex"]
    df["hasBird"] = labels["hasBird"]

    df.to_csv(interim_feats_dir + os.sep + file[:-4] + '_features.csv', index=False)

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

    # dataset to process
    dataset = config['general']["dataset"]

    # start logger
    logger = logging.getLogger(__name__)
    logger.info('creating features from ' + dataset)

    # input/output locations
    data_dir = 'data/processed/' + dataset
    label_loc = 'data/raw/' + dataset + '/metadata/labels.csv'
    interim_feats_dir = 'data/interim/' + dataset + '/features/'
    final_dir = 'data/features/' + dataset + os.sep
    env_dir = 'data/interim/' + dataset + '/env/' 

    if not os.path.isdir(interim_feats_dir):
        os.makedirs(interim_feats_dir, )
    if not os.path.isdir(final_dir):
        os.makedirs(final_dir, )
    if not os.path.isdir(env_dir):
        os.makedirs(env_dir, )

    # load config vars
    am_min_mod = config['AM']['min_mod']
    am_max_mod = config['AM']['max_mod']
    am_prominence_cutoff = config['AM']['prominence_cutoff']
    fm_window_length = config['FM']['window_length']
    fm_window_overlap = config['FM']['window_overlap']
    fm_threshold = config['FM']['threshold']

    # create and fill queue
    file_queue = multiprocessing.Queue()
    for file in os.listdir(data_dir):
        file_queue.put(file)

    pool = multiprocessing.Pool(None, runProcess, (file_queue,))
    pool.close()
    pool.join()

    csv_files = glob.glob(os.path.join(interim_feats_dir, "*.csv"))
    df_from_csv_files = (pd.read_csv(f, sep=',') for f in csv_files)
    df_merged = pd.concat(df_from_csv_files, ignore_index=True)
    df_merged = df_merged.dropna()
    df_merged = df_merged.drop_duplicates()
    df_merged.to_csv(final_dir + 'features.csv')