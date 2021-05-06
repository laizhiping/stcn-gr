### STCN-GR: Spatial-Temporal Convolutional Networks for Surface-Electromyography-Based Gesture Recognition

This repository contains the source codes of our work: *STCN-GR: Spatial-Temporal Convolutional Networks for Surface-Electromyography-Based Gesture Recognition*

#### Requirements
- PyTorch
- yaml
- tensorboard

#### Usage
##### 1. prepare dataset
The datasets and source codes should be organized by:
```
dataset
  - bandmyo
    - subject-1
      - session-1
        - gesture-1
          - trial-1.mat
          ...
  - capgmyo
stcn-gr
  - config
  - stcn
  - log
  - model
  - runs
  - run.py
```
in which the dataset file `.mat` includes the sEMG channel data which the shape of `num_frames x num_channels`. The folders `log`, `model` and `runs` are created  after running the source codes.

##### 2. train
run the command:
```
python run.py -cfg config/capgmyo.yaml -sg train 
```
##### 3. test
run the command:
```
python run.py -cfg config/capgmyo.yaml -sg test 
```