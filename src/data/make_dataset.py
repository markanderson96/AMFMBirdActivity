#!/bin/python

# tools
import logging
import os
import yaml
import sys
import time
import multiprocessing
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# modules for signal processing
import soundfile as sf
import numpy as np
import pandas as pd
from scipy import signal

def runProcess(file_queue):
    while not file_queue.empty():
        filename = file_queue.get(block=False)
        makeData(filename)
        time.sleep(0.1)

def makeData(file):
    logging.info('Pre-processing: ' + file)
    data, samplerate = sf.read(input_loc + file)
    data = np.asarray(data, dtype=np.float32)

    # filter
    b = signal.firwin(1024, [min_freq, max_freq], pass_zero=False, fs=samplerate)
    data = signal.filtfilt(b, 1.0, data)

    # Normalise again
    data += 1E-128 # avoid division by 0, arbitrarily small
    data /= np.max(np.abs(data))

    sf.write(output_loc + file, data, samplerate)       


if __name__ == '__main__':
    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    # useful to have project dir
    project_dir = Path(__file__).resolve().parents[2]

    # logger formatting
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
    logger.info('making final data set from raw data')

    # input/output locations
    input_loc = input_filepath + os.sep + dataset + '/wav/train/'
    output_loc = output_filepath + os.sep + dataset + os.sep
    if not os.path.isdir(output_loc):
        os.mkdir(output_loc)

    # load config vars
    min_freq = config['band_filter']["filt_min_freq"]
    max_freq = config['band_filter']['filt_max_freq']

    # create and fill queue
    file_queue = multiprocessing.Queue()
    for file in os.listdir(input_loc):
        file_queue.put(file)

    pool = multiprocessing.Pool(None, runProcess, (file_queue,))
    pool.close()
    pool.join()