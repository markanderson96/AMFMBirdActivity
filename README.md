AMFM_BAD
==============================

A Bird Activity Detector using lightweight ML methods and AM/Pitch based features
## AMFM Bird Activity Detector
A bird activity detector (BAD) utilising AM and Pitch based features.

## Motivation
Bird Activity Detection, has moved from traditional ML methods to approaches making use of CDNNs, RNNs and other computationally expensive architectures. The performance of such systems is impressive, but they cannot easily be scaled down for embedded applications, or low resource computing. 

Furthermore, the features used such as MFCCs, may not be the most suitable for usage in birdsong, as Mel based features are based on a human perceptual scale, tuned for voice. 

This project aims to be a framework that can be scaled down to embedded hardware, and to make use of alternative features. The result so far is a lightweight Random Forest Network, utilising only 8 features.

## Requirements
- [Numpy](https://github.com/numpy/numpy)
- [Pandas](https://github.com/pandas-dev/pandas)
- [pyfilterbank](https://github.com/SiggiGue/pyfilterbank)
- [SciPy](https://github.com/scipy/scipy)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [python-soundfile](https://github.com/bastibe/python-soundfile)


Project Organization
------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
