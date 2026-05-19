import os
import numpy as np
import pywt

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, welch
from pyunpack import Archive

from config import LABEL_MAPPING, SEGMENT_LENGTH


def extract_archive(base_path, extract_path):
    if not os.path.exists(extract_path) or not os.listdir(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        Archive(base_path).extractall(extract_path)
        print(f"Data extracted to {extract_path}")
    else:
        print(f"Data already exists in: {extract_path}")


def load_ecg_database(base_path):
    database = {"A": [], "N": [], "V": []}

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)
                main_folder = file_path.split("/")[-4]

                if main_folder in database:
                    try:
                        mat_data = loadmat(file_path)
                        database[main_folder].append((file, mat_data))
                    except Exception as error:
                        print(f"Error loading {file_path}: {error}")

    return database


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    b, a = butter(order, low, btype="low")
    return filtfilt(b, a, data)


def butter_highpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype="high")
    return filtfilt(b, a, data)


def wavelet_filter(signal, wavelet="db6", level=5):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(
            coeffs[i],
            value=np.std(coeffs[i]) * 0.5,
            mode="soft",
        )

    clean_signal = pywt.waverec(coeffs, wavelet)
    return clean_signal[:len(signal)]


def filter_ecg_signal(ecg_signal, sampling_rate):
    ecg_signal = wavelet_filter(ecg_signal, wavelet="db6", level=5)
    ecg_signal = butter_highpass_filter(ecg_signal, cutoff=0.5, fs=sampling_rate)
    ecg_signal = butter_lowpass_filter(ecg_signal, cutoff=50, fs=sampling_rate)
    return ecg_signal


def segment_signal_by_time(signal, sampling_rate, segment_duration=5, pad_last=False):
    segment_length = sampling_rate * segment_duration
    num_segments = len(signal) // segment_length

    segments = [
        signal[i * segment_length:(i + 1) * segment_length]
        for i in range(num_segments)
    ]

    if pad_last:
        remaining_samples = len(signal) % segment_length

        if remaining_samples > 0:
            last_segment = signal[-remaining_samples:]
            padding_length = segment_length - len(last_segment)
            last_segment = np.pad(last_segment, (0, padding_length), mode="constant")
            segments.append(last_segment)

    return np.array(segments)


def extract_features(segment, fs):
    peaks, properties = find_peaks(segment, height=0.5)
    rr_intervals = np.diff(peaks) / fs

    rMSSD = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else 0
    PRR50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals) if len(rr_intervals) > 1 else 0
    PRR20 = np.sum(np.abs(np.diff(rr_intervals)) > 0.02) / len(rr_intervals) if len(rr_intervals) > 1 else 0
    SDSD = np.std(np.diff(rr_intervals)) if len(rr_intervals) > 1 else 0
    SDRR = np.std(rr_intervals) if len(rr_intervals) > 0 else 0

    freqs, power = welch(segment, fs=fs)
    low_freq = np.sum(power[(freqs >= 0.1) & (freqs < 0.5)])
    mid_freq = np.sum(power[(freqs >= 0.5) & (freqs < 15)])
    high_freq = np.sum(power[(freqs >= 15) & (freqs < 40)])

    if len(rr_intervals) > 1:
        sd1 = np.std(np.diff(rr_intervals)) / np.sqrt(2)
        sd2 = np.std(rr_intervals) / np.sqrt(2)
    else:
        sd1, sd2 = 0, 0

    if len(rr_intervals) > 0:
        heart_rates = 60 / rr_intervals
        min_hr = np.min(heart_rates)
        max_hr = np.max(heart_rates)
        avg_hr = np.mean(heart_rates)
    else:
        min_hr, max_hr, avg_hr = 0, 0, 0

    peak_amplitudes = properties["peak_heights"] if "peak_heights" in properties else []
    avg_peak_amplitude = np.mean(peak_amplitudes) if len(peak_amplitudes) > 0 else 0

    if len(peaks) >= 5:
        pq_delay = (peaks[1] - peaks[0]) / fs
        qr_delay = (peaks[2] - peaks[1]) / fs
        rs_delay = (peaks[3] - peaks[2]) / fs
        st_delay = (peaks[4] - peaks[3]) / fs
    else:
        pq_delay, qr_delay, rs_delay, st_delay = 0, 0, 0, 0

    return [
        rMSSD, PRR50, PRR20, SDSD, SDRR,
        low_freq, mid_freq, high_freq,
        sd1, sd2,
        min_hr, max_hr, avg_hr,
        avg_peak_amplitude,
        pq_delay, qr_delay, rs_delay, st_delay,
    ]


def extract_ecg_signal(mat_data):
    if "ECG" not in mat_data:
        return None

    ecg_signal = mat_data["ECG"]

    if isinstance(ecg_signal, np.ndarray):
        if ecg_signal.ndim == 2 and ecg_signal.shape[1] == 1:
            ecg_signal = ecg_signal.flatten()

        if ecg_signal.dtype == "O":
            ecg_signal = ecg_signal[0].flatten()

    if not isinstance(ecg_signal, np.ndarray):
        return None

    if not np.issubdtype(ecg_signal.dtype, np.number):
        return None

    return ecg_signal.astype(float)


def preprocess_data(all_data, sampling_rate=400, segment_duration=5):
    X_segments = []
    X_features = []
    y = []

    for folder, files in all_data.items():
        if folder not in LABEL_MAPPING:
            continue

        label = LABEL_MAPPING[folder]
        print(f"Processing folder: {folder}, files: {len(files)}")

        for mat_file_name, mat_data in files:
            try:
                ecg_signal = extract_ecg_signal(mat_data)

                if ecg_signal is None:
                    print(f"Invalid ECG data in file: {mat_file_name}")
                    continue

                ecg_signal = filter_ecg_signal(ecg_signal, sampling_rate)

                augmented_signals = [ecg_signal]

                if label == 0:
                    augmented_signals.append(-ecg_signal)

                for aug_signal in augmented_signals:
                    segments = segment_signal_by_time(
                        aug_signal,
                        sampling_rate,
                        segment_duration,
                        pad_last=False,
                    )

                    for segment in segments:
                        features = extract_features(segment, fs=sampling_rate)

                        X_segments.append(segment)
                        X_features.append(features)
                        y.append(label)

            except Exception as error:
                print(f"Error processing {mat_file_name}: {error}")

    X_segments = np.array(X_segments)
    X_features = np.array(X_features)
    y = np.array(y)

    X_segments = X_segments.reshape(X_segments.shape[0], SEGMENT_LENGTH, 1)

    return X_segments, X_features, y
