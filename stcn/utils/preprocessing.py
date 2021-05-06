import scipy.signal
import numpy as np
from scipy.ndimage.filters import median_filter

# default: ninapro
# f: cut-off frequency  fs:sampling frequency
def lpf(x, f=1, fs=100):
    # return x
    wn = 2.0 *f / fs
    b, a = scipy.signal.butter(1, wn, 'low')
    x = np.abs(x)
    output = scipy.signal.filtfilt(
        b, a, x, axis=0,
        padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)
    )
    # 濾波之後可能索引倒排，需要copy
    return output.copy()

# default: cslhdemg
def bpf(x, order=4, cut_off=[20, 400], sampling_f = 2048):
    # return x
    wn = [2.0 * i / sampling_f for i in cut_off]
    b, a = scipy.signal.butter(order, wn, "bandpass")
    output = scipy.signal.filtfilt(b, a, x)
    # print(x.shape, output.shape)
    return output.copy()

def butter_bandpass_filter(x, order=4, lowcut=20, highcut=400, fs = 2048):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
    y = scipy.signal.lfilter(b, a, x)
    return y


def butter_bandstop_filter(x, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = scipy.signal.butter(order, [low, high], btype='bandstop')
    y = scipy.signal.lfilter(b, a, x)
    return y


def butter_lowpass_filter(x, order=1, cut=1, fs=100, zero_phase=False):
    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = scipy.signal.butter(order, cut, btype='low')
    y = (scipy.signal.filtfilt if zero_phase else scipy.signal.lfilter)(b, a, x)
    return y


def amplify(x, rate=1000):
    return x * rate

def u_normalization(x, u):
    return np.sign(x) * np.log(1 + u*np.abs(x)) / np.log(1 + u)

def continuous_segments(label):
    label = np.asarray(label)

    if not len(label):
        return

    breaks = list(np.where(label[:-1] != label[1:])[0] + 1) # 找出每段的分界点
    # print(label, breaks)
    for begin, end in zip([0] + breaks, breaks + [len(label)]): # 构造段 [begin, end)
        assert begin < end
        yield begin, end

# for csl-hdemg:  x: (6144, 168)
def csl_cut(x):
    window = 150
    last = x.shape[0] // window * window
    # print(x[:last].shape)
    new_x = x[:last].reshape((-1, window, x[:last].shape[1])) # (40, 150, 168)
    # print(new_x.shape)
    rms = np.sqrt(np.mean(np.square(new_x), axis=1)) # (40, 168)
    # print(rms.shape)
    rms = [median_filter(image, 3).ravel() for image in rms.reshape(-1, 24, 7)] # (40, 168)
    # rms = [scipy.signal.medfilt(i, 3) for i in rms] # (40, 168)
    # print(len(rms), rms[0].shape)
    rms = np.mean(rms, axis=1) # (40,)
    threshold = np.mean(rms)
    mask = rms > threshold # (40,)
    # print(mask.shape)
    for i in range(1, len(mask) - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    begin, end = max(continuous_segments(mask),
                     key=lambda s: (mask[s[0]], s[1] - s[0])) # 比较规则，段起始是true，并且长度最长

    begin = begin * window
    end = end * window
    return x[begin:end]

def median_filt(x, order=3):  # x: (n, 168)
    return np.array([median_filter(image, 3).ravel() for image
                    in x.reshape(-1, 24, 7)]).astype(np.float32)


    new_x = x.reshape(-1, 24, 7)
    for i in range(new_x.shape[0]):
        new_x[i] = median_filter(new_x[i], 3)
    new_x = new_x.reshape(-1, 168)
    return new_x
    return scipy.signal.medfilt(x, (1,3))

def abs(x):
    return np.abs(x)

def downsample(x, rate):
    return x[::rate].copy()