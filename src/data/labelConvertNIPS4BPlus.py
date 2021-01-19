#!/bin/python

import re
import os
import logging
import yaml
import pandas as pd
import numpy as np
import soundfile as sf
from math import floor, ceil
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def lengthAdjust():
    logger = logging.getLogger(__name__)
    logger.info('adjusting length for label congruity')

    data_dir = 'data/raw/nips4b/wav/train/'
    out_dir = 'data/interim/nips4b/wav/train/'
    
    # for filename in os.listdir(data_dir):
    #     [data, samplerate] = sf.read(data_dir + filename)
    #     n = data.size // samplerate
    #     m = data.size % samplerate
    #     if m < (samplerate / 2):
    #         n = n + 0.5
    #     else:
    #         n = n + 1
    #     if n > N:
    #         N = n
    for filename in os.listdir(data_dir):
        [data, samplerate] = sf.read(data_dir + filename)
        desired_length = int(samplerate * N)
        logger.info('Processing ' + filename)
        logger.info("Number of Samples: " + str(desired_length))
        logger.info("File length: " + str(desired_length/samplerate) + "s")
        data = np.asarray(data)
        zero_pad_length = desired_length - data.size
        zero_pad = np.zeros(zero_pad_length)
        data = np.append(data, zero_pad)
        sf.write(out_dir + filename, data, samplerate)

def labelConvert():
    logger = logging.getLogger(__name__)
    logger.info('converting nips4b labels')

    label_dir = 'data/raw/nips4b/metadata/'
    data = []
    for filename in os.listdir(label_dir):
        if filename == 'labels.csv':
            continue
        logger.info('processing: ' + filename)
        df = pd.read_csv(label_dir + filename, header=None, names=['start_time', 'duration', 'hasBird'])
        if df.empty:
            continue

        # change duration to end_time, species to just hasBird
        end_time = df['start_time'] + df['duration']
        df['end_time'] = end_time
        df['hasBird'] = 1
        df = df.drop(['duration'], axis=1)
        df = df.reindex(columns=['start_time', 'end_time', 'hasBird'])

        # add hasBird 0
        noBird = np.zeros(df.shape)
        i = 1
        while i + 1 <= len(noBird):
            noBird[i, 0] = df.iloc[i - 1, 1]
            noBird[i, 1] = df.iloc[i, 0]
            i += 1

        # merge with original data
        end_time = df.iloc[i-1, 1]
        noBird = np.append(noBird, [[end_time, 5.5, 0]], axis=0)
        noBird = pd.DataFrame(noBird, columns=['start_time', 'end_time', 'hasBird'])    
        df = df.append(noBird)
        df = df.sort_values(by='start_time')

        # round to nearest 0.5
        df['start_time'] = df['start_time'].apply(lambda x: np.round(x * 2) / 2.0)
        df['end_time'] = df['end_time'].apply(lambda x: np.round(x * 2) / 2.0)
        
        # clean up
        df = df.drop_duplicates(subset=['start_time', 'end_time'])
        dup_indices = df[(df['start_time'] == df['end_time'])].index
        df = df.drop(dup_indices)
        df = df.reset_index(drop=True)


        file_id = re.findall(r'-?\d+?\d*', filename)[0]

        t = 0.0
        i = 0
        j = 0
        while t + 1.0 <= N:
            _start_time = df.iloc[i, 0]
            end_time = df.iloc[i, 1]
            hasBird = df.iloc[i, 2]

            if (t + 1.0) <= end_time:
                data.append(['nips4b_birds_trainfile' + file_id + '-' + str(j), hasBird])
                t = t + 0.5
                j = j + 1
            else:
                i = i + 1

    df = pd.DataFrame(data=data, columns = ['fileIndex', 'hasBird'])
    df.to_csv('data/raw/nips4b/metadata/labels.csv', index=False)

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

    N = 6

    lengthAdjust()
    labelConvert()