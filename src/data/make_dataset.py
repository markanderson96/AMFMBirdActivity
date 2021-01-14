# -*- coding: utf-8 -*-
import click
import logging
import os
import yaml
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from scipy import signal

from activitydetector import AD 
from noisereduce import noiseReduction


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    dataset = config['general']["dataset"]

    for file in os.listdir(input_filepath):
        logging.info('Pre-processing: ' + file)
        data, samplerate = sf.read(input_filepath + '/' + file)
        data = np.asarray(data, dtype=np.float32)

        # preprocessing
        # activity detector
        ad_window_length = config['activity_detector']['window_length']
        ad_window_overlap = config['activity_detector']['window_overlap']
        ad_block_size = config['activity_detector']['block']
        ad_threshold = config['activity_detector']['threshold']

        ad = AD(data, samplerate, window_length=ad_window_length,
                window_overlap=ad_window_overlap, block_size=ad_block_size,
                threshold=ad_threshold)
        data = ad.reconstruct()

        # noise reduction
        nr_window_size = config['noise_reduce']['window_length']
        nr_overlap = config['noise_reduce']['overlap']
        nr_nth_oct = config['noise_reduce']['nth_oct']
        nr_norm_freq = config['noise_reduce']['norm_freq']
        nr_start_band = config['noise_reduce']['start_band']
        nr_end_band = config['noise_reduce']['end_band']
        nr_r_filters = config['noise_reduce']['r_filters']

        nr = noiseReduction(samplerate=samplerate, window_size=nr_window_size,
                            window_overlap=nr_overlap, nth_oct=nr_nth_oct,
                            norm_freq=nr_norm_freq, start_band=nr_start_band,
                            end_band=nr_end_band, r_filters=nr_r_filters)
        data = nr.noiseReduction(data)

        # filter
        min_freq = config['band_filter']["filt_min_freq"]
        max_freq = config['band_filter']['filt_max_freq']

        b = signal.firwin(128, [min_freq, max_freq], pass_zero=False, fs=samplerate)
        data = signal.filtfilt(b, 1.0, data)

        # normalise the audio
        data += 1E-128 # avoid division by 0, arbitrarily small
        data /= np.max(np.abs(data))

        sf.write(output_filepath + '/' + dataset + '/' + file, data, samplerate)


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