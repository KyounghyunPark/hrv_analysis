import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import label
from scipy.stats import zscore

# path setting
project_path = os.path.join(os.getcwd(), os.pardir)
data_path = os.path.join(project_path, 'data')
results_path = os.path.join(project_path, 'results')


def get_plot_ranges(start=10, end=20, n=5):
    '''
    >> list(get_plot_ranges(0, 10, 2))
    >>[(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

    >> list(get_plot_ranges(0, 10, 3))
    >> [(0, 3), (3, 6), (6, 9), (9, 10)]
    '''    
    distance = end - start
    for i in np.arange(start, end, np.floor(distance/n)) :
        yield (int(i), int(np.minimum(end, np.floor(distance/n)+i)))



def detect_peaks(ecg_signal, threshold=0.3, qrs_filter=None):
    '''
    Peak detection algorithm using cross correlation and threshold
    '''
    if qrs_filter is None:
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)

    # normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
        
    # calculate cross correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # return peaks (values in ms) using threshold
    return ecg_signal[similarity > threshold].index, similarity


def group_peaks(p, threshold=5) :
    output = np.empty(0)
    
    # label groups of sample that belong to the same peak
    peak_groups, num_groups = label(np.diff(p) < threshold)

    # iterate through groups and take the mean as peak index
    for i in np.unique(peak_groups)[1:]:
        peak_group = p[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))

    return output
    



# load data
df = pd.read_csv(os.path.join(data_path, "ecg.csv"), sep=';', index_col="ms")


# sample frequency
settings = {}
settings['fs'] = 500

'''
# explore data
plt.figure(figsize=(40, 10))
start=0
stop=5000
duration = (stop-start) / settings['fs']   # 10초(10,000 ms) 크기의 데이터 출력 (5,000 = 500 samples/s * 10s)
plt.title("ECG Signal, slice of %.1f seconds" % duration, fontsize=24)
plt.plot(df[start:stop].index, df[start:stop].heartrate, color='#51A6D8', linewidth=1)
plt.xlabel("Time (ms)", fontsize=16)
plt.ylabel("Amplitude (arbitrary unit)", fontsize=16)
plt.show()
'''

# find RR-intervals
sample_from = 60000
sample_to = 70000
nr_plots = 1

for start, stop in get_plot_ranges(sample_from, sample_to, nr_plots):    
    # get slice data of ECG data
    cond_slice = (df.index >= start) & (df.index < stop)
    ecg_slice = df.heartrate[cond_slice]
    
    # detect peaks
    peaks, similarity = detect_peaks(ecg_slice, threshold=0.3)

    '''    
    # plot similarity
    plt.figure(figsize=(40,20))

    plt.subplot(211)
    plt.title("ECG Signal with found peaks", fontsize=24)
    plt.plot(ecg_slice.index, ecg_slice, label="ECG", linewidth=1)
    plt.plot(peaks, np.repeat(600, peaks.shape[0]), markersize=10, label="peaks", color="orange", marker="o", linestyle="None")
    plt.legend(loc="upper right", fontsize=20)
    plt.xlabel("Time (milliseconds)", fontsize=16)
    plt.ylabel("Amplitude (arbitrary unit)", fontsize=16)

    plt.subplot(212)
    plt.title("Similarity with QRS template", fontsize=24)
    plt.plot(ecg_slice.index, similarity, label="Similalrity with QRS filter", color="olive", linewidth=1)
    plt.legend(loc="upper right", fontsize=20)
    plt.xlabel("Time (milliseconds)", fontsize=16)
    plt.ylabel("Similarity (normalized)", fontsize=16)

    plt.savefig(os.path.join(results_path, "peaks-%s-%s.png" % (start, stop)))

    plt.show()
    '''


# detect peaks
peaks, similarity = detect_peaks(df.heartrate, threshold=0.3)

# group peaks
grouped_peaks = group_peaks(peaks)

'''
# plot peaks
plt.figure(figsize=(40, 10))
plt.title("Group similar peaks together", fontsize=24)
plt.plot(df.index, df.heartrate, label="ECG", linewidth=1)
plt.plot(peaks, np.repeat(600, peaks.shape[0]), markersize=10, label="peaks", color="orange", marker='o', linestyle="None")
plt.plot(grouped_peaks, np.repeat(620, grouped_peaks.shape[0]), markersize=13, label="grouped peaks", color='k', marker="v", linestyle="None")
plt.legend(loc="upper right", fontsize=20)
plt.xlabel("Time (ms)", fontsize=16)
plt.ylabel("Amplitude (arbitrary unit)", fontsize=16)
plt.gca().set_xlim(0, 200)

plt.show()
'''


# detect peaks
peaks, similarity = detect_peaks(df.heartrate, threshold=0.3)

# group peaks so we get a single peak per beat 
grouped_peaks = group_peaks(peaks)

# RR-intervals are the differences between successive peaks
rr = np.diff(grouped_peaks)

'''
# plot RR-intervals
plt.figure(figsize=(40, 15))
plt.title("RR-intervals", fontsize=24)
plt.xlabel("Time (ms)", fontsize=16)
plt.ylabel("RR-interval (ms)", fontsize=16)
plt.plot(np.cumsum(rr), rr, label="RR-interval", linewidth=2)

plt.show()
'''

# artifiact removal
'''

plt.figure(figsize=(40,10))
plt.title("Distribution of RR-intervals", fontsize=24)
sns.kdeplot(rr, label="rr-intervals", shade=True)

outlier_low = np.mean(rr)-2 * np.std(rr)
outlier_high = np.mean(rr)+2 * np.std(rr)

plt.axvline(x=outlier_low)
plt.axvline(x=outlier_high, label="outlier boundary")
plt.text(outlier_low - 270, 0.004, "outlier low (<mena -2 sigma)", fontsize=20)
plt.text(outlier_high+20, 0.004, "outliers high (> mean + 2 sigma)", fontsize=20)

plt.xlabel("RR-interval (ms)", fontsize=16)
plt.ylabel("Density", fontsize=16)

plt.legend(fontsize=24)


plt.show()
'''


'''
plt.figure(figsize=(40,15))

rr_corrected = rr.copy()
rr_corrected[np.abs(zscore(rr)) > 2] = np.median(rr)

plt.title("RR-intervals", fontsize=24)
plt.xlabel("Time (ms)", fontsize=16)
plt.ylabel("RR-interval (ms)", fontsize=16)

plt.plot(rr, color='red', linewidth=1, label="RR-intervals")
plt.plot(rr_corrected, color='green', linewidth=2, label="RR-intervals aftger correction")
plt.legend(fontsize=20)

plt.show()
'''

# plot ECG vs. RR interval
sample_from = 200000
sample_to = 300000
nr_plots = 10

#detect peaks
peaks, similarity = detect_peaks(df.heartrate, threshold=0.3)

#group peaks so we get a single peak per beat
grouped_peaks = group_peaks(peaks)

# RR-intervals are the differences between successive peaks
rr = np.diff(grouped_peaks)

'''
for start, stop in get_plot_ranges(sample_from, sample_to, nr_plots):
    plt.figure(figsize=(40, 20))

    plt.title("ECG signal & RR-intervals", fontsize=24)
    plt.plot(df.index, df.heartrate, label="ECG", linewidth=1)
    plt.plot(grouped_peaks, np.repeat(600, grouped_peaks.shape[0]), markersize=14, label="Found peaks", color="orange", marker='o', linestyle="None")
    plt.legend(loc="upper left", fontsize=20)
    plt.xlabel("Time (milliseconds)", fontsize=16)
    plt.ylabel("Amplitude (arbitrary unit)", fontsize=16)
    plt.gca().set_ylim(400, 800)

    ax2 = plt.gca().twinx()
    ax2.plot(np.cumsum(rr)+peaks[0], rr, label="RR-intervals", color="#A561D8", linewidth=2, markerfacecolor="#A651D8", markeredgewidth=0, marker='o', markersize=18)
    ax2.set_xlim(start, stop)
    ax2.set_ylim(-2000, 2000)
    ax2.legend(loc="upper right", fontsize=20)

    plt.xlabel("Time (ms)", fontsize=16)
    plt.ylabel("RR-intervals (ms)", fontsize=16)

    plt.savefig(os.path.join(results_path, "ecg-with-rr-%s-%s.png" % (start, stop)))
    
plt.show()
'''

# export rr-intervals for analysis in Kubios
rr_corrected = rr.copy()
rr_corrected[np.abs(zscore(rr)) > 2] = np.median(rr)
np.savetxt(os.path.join(results_path, "rr.txt"), rr_corrected, fmt='%d')

# plot ECG vs. maually corrected RR intervals
rr_manual = np.loadtxt(os.path.join(data_path, "manual-correction-rr.txt"), dtype=int)