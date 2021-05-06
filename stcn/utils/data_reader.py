import torch
import os
import scipy.io
import numpy as np
from . import preprocessing
from . import data_augmentation as da

class DataReader(torch.utils.data.Dataset):
    def __init__(self, subjects, sessions, gestures, trials, args):
        super(DataReader, self).__init__()
        self.subjects = subjects
        self.sessions = sessions
        self.gestures = gestures
        self.trials = trials
        self.args = args
        self.load_dataset()
        # print("Finish load data")
        self.preprocess()
        # print("Finish preprocess")
        # self.augment()
        # print("Finish augment")
        self.generate()
        # print("Finish generate")
        self.shuffle()
        # for i in range(len(self.X)):
        #     print(self.X[i].shape, self.y[i])
        # print("Finish shuffle")

        # self.X: --list, list of all trials, self.X[i]: np.ndarray(sample_point, channel)
        # self.y: --list, list of gestures of all trials
        # self.x_offsets: --list of all windows, self.x_offset[i]: tuple(index_in_self.X, window_start)
        # self.indexes: --list, indexes of self.x_offset

    def __getitem__(self, i):
        idx = self.indexes[i]
        trial_and_window = self.x_offsets[idx]
        # print(trial_and_window)

        trial_index = trial_and_window[0]
        windows_start = trial_and_window[1]
        window_size = self.args.window_size
        train_data = self.X[trial_index]
        if window_size != 0: # 窗口大小不为0取窗口大小
            x = train_data[windows_start:windows_start + window_size, :]  # (frame, channel)
        else:
            x = train_data
        y = self.y[trial_index]

        x = np.expand_dims(x, axis=0) # (1, frame, channel)



        if self.args.dataset_name in ["capgmyo", "bandmyo", "ninapro"]:
            y = y - 1
        # print(x.shape)
        return x, y

    def __len__(self):
        return len(self.indexes)

    def init_dataset(dataset_list):
        return zip(*dataset_list)

    def shuffle(self):
        np.random.shuffle(self.indexes)

    def load_dataset(self):
        X, y = [], []
        root = self.args.dataset_path

        for subject in self.subjects:
            for session in self.sessions:
                for gesture in self.gestures:
                    if self.args.dataset_name == "cslhdemg" and gesture == 0:
                        idle_trs = [2,4,7,8,11,13,19,25,26,30]
                        trs = [idle_trs[i-1] for i in self.trials]
                    else:
                        trs = self.trials

                    # print(gesture, len(trs))
                    for trial in trs:
                        # cslhdemg subject 4 session 4 gesture 4,5 只有9个trials
                        if self.args.dataset_name and subject == 4 and session == 4 and (gesture == 8  or gesture == 9) and trial == 10:
                            continue
                        path = os.path.join(root, f"subject-{subject}", f"session-{session}", f"gesture-{gesture}", f"trial-{trial}")
                        # print(path)
                        mat = scipy.io.loadmat(path)
                        # print(mat["emg"].shape)
                        X.append(mat["emg"])
                        y.append(gesture)

                # print(f"Finish load subject {subject} session {session}")
        # print(len(X)) # 270
        self.X = X
        self.y = y

    def preprocess(self):
        window_size = self.args.window_size
        window_step = self.args.window_step
        num_trails = len(self.X)

        if self.args.dataset_name == "cslhdemg":
            for i in range(num_trails):
                plan = 1
                if plan == 1:
                    # plan 1
                    self.X[i] = preprocessing.butter_bandpass_filter(self.X[i])
                    self.X[i] = preprocessing.csl_cut(self.X[i])
                    self.X[i] = preprocessing.median_filt(self.X[i])
                elif plan == 2:
                    # plan 2
                    self.X[i] = preprocessing.csl_cut(self.X[i])
                    self.X[i] = preprocessing.abs(self.X[i])
                    self.X[i] = preprocessing.butter_lowpass_filter(self.X[i], order=1, cut=1, fs=2048, zero_phase=True)
                    self.X[i] = preprocessing.downsample(self.X[i], 5)
                elif plan == 3:
                    # plan 3
                    self.X[i] = preprocessing.csl_cut(self.X[i])
                    self.X[i] = preprocessing.abs(self.X[i])
                    self.X[i] = preprocessing.butter_lowpass_filter(self.X[i], order=1, cut=1, fs=2048, zero_phase=True)

        elif self.args.dataset_name == "ninapro":
            for i in range(num_trails):
                self.X[i] = preprocessing.butter_lowpass_filter(self.X[i])


        # for i in range(num_trails):
        #     if self.config["dataset"]["lpf"]:
        #         self.X[i] = preprocessing.lpf(self.X[i])
        #     if self.config["dataset"]["bpf"]:
        #         # self.X[i] = preprocessing.bpf(self.X[i])
        #         self.X[i] = preprocessing.butter_bandpass_filter(self.X[i])

        #     if self.config["dataset"]["abs"]:
        #         self.X[i] = np.abs(self.X[i])
        #     # print(self.X[i].shape)
        #     if self.config["dataset"]["move_average"]: # move average
        #         self.X[i] = np.apply_along_axis(
        #             lambda m: np.convolve(m, np.ones((window_step,)) / window_step, mode='same'), axis=0, arr=self.X[i])
        #     # print(self.X[i].shape)

        # if self.config["dataset"]["amplify"]:
        #     self.x[i] = preprocessing.amplify(self.x[i])

    def generate(self):
        # self.__augment()
        self.make_segments()
        self.indexes = np.arange(len(self.x_offsets))

    def make_segments(self):
        window_size = self.args.window_size
        window_step = self.args.window_step
        x_offsets = []
        if window_size != 0:
            for i in range(len(self.X)):
                trial_data = self.X[i]  # (num_samples, channels)
                num_samples = trial_data.shape[0]
                # print(trial_data.shape)
                for j in range(0, num_samples - window_size, window_step):
                    x_offsets.append((i, j))
        else: # 窗口大小为0时起始位置为0
            x_offsets = [(i, 0) for i in range(len(self.X))]

        self.x_offsets = x_offsets


    def augment(self):
        X_aug, y_aug = [], []

        self.size_factor = 0
        self.time_warping = 0.2
        self.mag_warping = 0.2
        self.noise_snr_db = 25
        self.permutation = 0
        self.rotation = 0
        self.rotation_mask=None
        self.scale_sigma = 0

        for i in range(len(self.X)):
            for _ in range(self.size_factor):
                x = np.copy(self.X[i])
                if self.permutation != 0:
                    x = da.permute(x, nPerm=self.permutation)
                if self.rotation != 0:
                    x = da.rotate(x, rotation=self.rotation, mask=self.rotation_mask)
                if self.time_warping != 0:
                    x = da.time_warp(x, sigma=self.time_warping)
                if self.scale_sigma != 0:
                    x = da.scale(x, sigma=self.scale_sigma)
                if self.mag_warping != 0:
                    x = da.mag_warp(x, sigma=self.mag_warping)
                if self.noise_snr_db != 0:
                    x = da.jitter(x, snr_db=self.noise_snr_db)

                if self.permutation or self.rotation or self.time_warping or self.scale_sigma or self.mag_warping or self.noise_snr_db:
                    X_aug.append(x)
                    y_aug.append(self.y[i])

            X_aug.append(self.X[i])
            y_aug.append(self.y[i])

        self.X = X_aug
        self.y = y_aug


