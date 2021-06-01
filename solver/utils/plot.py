import os
import json
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
import preprocessing

def diff_gestures(name, root, gestures, subject=1, session=1, trial=1):

    show_gestures = gestures[2:3]
    show_channels = range(120, 128)

    gesture_data = {}
    abs_max = 0.0
    abs_min = 0.0
    for gesture in show_gestures: # 显示的姿势
        path = os.path.join(root, f"subject-{subject}", f"session-{session}", f"gesture-{gesture}", f"trial-{trial}.mat")
        mat = scipy.io.loadmat(path)
        # print(mat.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'emg'])
        x = mat["emg"]
        # print(x.shape) # (6144, 168)

        # cslhdemg预处理
        x = preprocessing.butter_bandpass_filter(x)
        begin, end = preprocessing.csl_cut(x, 150)
        # print(begin, end)
        x = x[begin:end]
        x = preprocessing.median_filt(x)

        abs_min = np.abs(x).min() if np.abs(x).min() < abs_min else abs_min
        abs_max = np.abs(x).max() if np.abs(x).max() > abs_max else abs_max
        gesture_data[gesture] = x

    # print(abs_min, abs_max) # 0.0 0.06244087038789026


    fig, ax = plt.subplots(nrows=len(show_channels), ncols=1)
    plt.suptitle(f"{name} subject {subject} session {session} trial {trial} gesture {show_gestures} channel {show_channels[0]}~{show_channels[-1]}")

    for channel in show_channels:
        # rx = range(end-begin)
        # ry = [0.001]*(end-begin)
        # ax[channel].plot(rx, ry)
        for gesture, x in gesture_data.items():
           ax[channel%(len(show_channels))].plot(x[:, channel])
           # ax[channel].set_title(f"channel {channel}")

    plt.show()


if __name__ == "__main__":
    config_path = "../../config/cslhdemg.json"
    # config_path = "../../config/capgmyo.json"
    with open(config_path) as f:
        config = json.load(f)

    name = config["dataset"]["name"]
    root = os.path.join("..", config["dataset"]["path"])

    gestures = config["dataset"]["gestures"]
    diff_gestures(name, root, gestures)