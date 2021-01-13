#!/bin/python

import re
import os
import logging
import yaml
import pandas as pd
from math import floor, ceil
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def parse(filepath):
    """
    Parse Praat Text at a given filepath

    Parameters
    ----------
    filepath : str
        Filepath for file to be parsed

    Returns
    -------
    data : pd.DataFrame
        Data Frame of Parsed data
    """

    xmin_list = []
    xmax_list = []
    text_list = []
    with open(filepath, 'r') as file:
        line = next(file)
        while line:
            line = ''.join(line.split())
            reg_match = _RegExLib(line)

            if reg_match.xmin:
                xmin = float(reg_match.xmin.group(1))
                xmin = round(xmin * 2) / 2.0
                xmin_list.append(xmin)

            if reg_match.xmax:
                xmax = float(reg_match.xmax.group(1))
                xmax = round(xmax * 2) / 2.0
                xmax_list.append(xmax)

            if reg_match.text:
                text = int(reg_match.text.group(1))
                text_list.append(text)
            
            line = next(file, None)

    xmin_list = xmin_list[2:]
    xmax_list = xmax_list[2:]
    data = {
        'xmin' : xmin_list,
        'xmax' : xmax_list,
        'text' : text_list
    }
    data = pd.DataFrame(data)

    return data

class _RegExLib:
    """Set up RegEx"""
    _reg_xmin = re.compile('xmin=(.*)')
    _reg_xmax = re.compile('xmax=(.*)')
    _reg_text = re.compile('text="(.*)"')

    __slots__ = ['xmin', 'xmax', 'text']

    def __init__(self, line):
        self.xmin = self._reg_xmin.match(line)
        self.xmax = self._reg_xmax.match(line)
        self.text = self._reg_text.match(line)

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # load config file
    with open(str(project_dir) + '/config/config.yaml') as file:
        config = yaml.safe_load(file)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    data = []

    for filename in os.listdir('data/external/praat'):
        if (filename.endswith('.Collection')):
            continue
        
        logger.info('processing: ' + filename)
        praat_data = parse('data/external/praat/' + filename)
        t = 0.0
        i = 0
        j = 0
        while t + 1.0 <= 10.0:
            xmin = praat_data.iloc[i, 0]
            xmax = praat_data.iloc[i, 1]
            label = praat_data.iloc[i, 2]
            
            if (t + 0.5) <= xmax:
                data.append([filename.split('.')[0] + '_' + str(j), label])
                t = t + 0.5
                j = j + 1
            else:
                i = i + 1
    
    df = pd.DataFrame(data=data, columns = ['fileIndex', 'hasBird'])
    df.to_csv('data/raw/ff1010/metadata/labels.csv', index=False)