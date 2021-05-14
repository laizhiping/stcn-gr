### STCN-GR: Spatial-Temporal Convolutional Networks for Surface-Electromyography-Based Gesture Recognition

This repository contains the source codes of our work: *STCN-GR: Spatial-Temporal Convolutional Networks for Surface-Electromyography-Based Gesture Recognition*.

**Before running, please install Anaconda.**

#### 1. Install requirements
```
pip install -r requirements.txt
```

##### 2. Prepare datasets
The datasets and source codes should be organized by:
```
dataset
  - capgmyo
    - subject-1
      - session-1
        - gesture-1
          - trial-1.mat
          ...
  - bandmyo
stcn-gr
  - config
  - stcn
  - log
  - model
  - runs
  - run.py
```
The dataset file *.mat* includes the sEMG channel data and the shape is *num_frames \* num_channels*. The folders *log*, *model* and *runs* are generated by codes.

##### 3. Train/Test
Please run the command:
```
python run.py -cfg config/capgmyo.yaml -sg train [options]
```
or
```
python run.py --config config/capgmyo.yaml --stage train [options]
```

*-cfg/--config* indicates the path of configuration file, *-sg/--stage* indicates stage (**train** or **test**), *options* can be *-s/-ne/-wz/-ws/-bs*, and more details can be seen in *run.py*.