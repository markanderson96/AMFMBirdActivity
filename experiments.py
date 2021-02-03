#!/bin/python

import os
import yaml
import smtplib
import pyDOE2 as pyDOE
import numpy as np
import pandas as pd
 
def main():
    ad_threshold_values = [0.3, 0.35, 0.4]
    nr_r_filters_values = [3, 6]
    am_min_mod_values = [1, 2]
    am_max_mod_values = [10, 8]
    am_prom_cut_values = [2, 3, 4]
    am_depth_t_values = [0.005, 0.01]
    fm_threshold_values = [0.2, 0.3]

    variables_list = [
        ad_threshold_values,
        nr_r_filters_values,
        am_min_mod_values,
        am_max_mod_values,
        am_prom_cut_values,
        am_depth_t_values,
        fm_threshold_values
    ]

    experiment_levels = pyDOE.fullfact([3, 2, 2, 2, 3, 2, 2])
    experiment_table = np.empty(experiment_levels.shape)

    for col in range(experiment_levels.shape[1]):
        for row in range(experiment_levels.shape[0]):
            level_index = experiment_levels[row, col]
            experiment_table[row, col] = variables_list[col][int(level_index)]

    with open('results.csv', 'w') as csv_file:
        csv_file.write('rf,svm,combo\n')
    csv_file.close()

    for row in range(experiment_levels.shape[0]):
        with open('config/config.yaml', 'w') as config_file:
            config_file.write(template.format(*experiment_table[row, :]))
        config_file.close()

        if os.system('python src/features/build_features.py'):
            break
        if os.system('python src/models/train_model.py'):
            break
        if row+1 % 26 == 0:
            mail('Experiment is {}% Complete'.format(float(row/432)))
            continue

    mail('Experiment Complete!')
    df = pd.DataFrame(experiment_table, columns=['ad_threshold', 'nr_r_filters', 
                                                 'am_min', 'am_max', 'am_prom',
                                                 'am_depth', 'fm_threshold'])
    df.to_csv('levels.csv')

def mail(body):
    # Login with your Gmail account using SMTP
    smtp_server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    smtp_server.login(sender, password)
    
    # Let's combine the subject and the body onto a single message
    message = f"Subject: {subject}\n\n{body}"
    
    # We'll be sending this message in the above format (Subject:...\n\nBody)
    smtp_server.sendmail(sender, receiver, message)
    
    # Close our endpoint
    smtp_server.close()

if __name__ == "__main__":
    template = """
# [general]
general:
  dataset: nips4b

# [classifier settings]
classifier:
  save_model: True
  test_size: 0.2
  shuffle: True
  random_state: 0
  verbose: 0
  scoring: f1

random_forest:
  n_estimators: 500
  criterion: entropy
  max_depth: 8
  min_samples_leaf: 8
  n_jobs: -1

svm:
  kernel: linear
  degree: 2 
  gamma: auto
  probability: True
  max_iter: -1

# [preprocessing]
activity_detector:
  window_length: 0.1
  window_overlap: 0.05
  block: 0.25
  threshold: {}
  band_start: 800
  band_end: 16000

noise_reduce:
  window_length: 0.1
  overlap: 0.05
  nth_oct: 6
  norm_freq: 2000
  start_band: -4
  end_band: 15
  r_filters: {}

band_filter:
  filt_min_freq: 800
  filt_max_freq: 16000

# Features
AM:
  min_mod: {}
  max_mod: {}
  prominence_cutoff: {}
  depth_threshold: {}

FM:
  window_length: 0.02
  window_overlap: 0.01
  threshold: {} """

    sender = 'andersm3burner@gmail.com'
    password = "small batch irish craft beer"
    subject = 'Preprocessing Hyperparameters Experiment'

    receiver = 'andersm3@tcd.ie'

    # Endpoint for the SMTP Gmail server (Don't change this!)
    smtp_server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    mail('Starting Experiment')
    main()