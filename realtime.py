import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from config import (
    SAMPLING_RATE,
    SEGMENT_DURATION,
    SEGMENT_LENGTH,
    CLASS_LABELS,
    MODEL_PATH,
    SCALER_PATH,
)

from model import Attention
from preprocessing import (
    filter_ecg_signal,
    segment_signal_by_time,
    extract_features,
)


def load_saved_model_and_scaler():
    model = load_model(
        MODEL_PATH,
        custom_objects={"Attention": Attention},
    )
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def process_real_time_signal(
    signal,
    model,
    scaler,
    sampling_rate=SAMPLING_RATE,
    segment_duration=SEGMENT_DURATION,
    show_plot=True,
):
    ecg_signal = filter_ecg_signal(signal, sampling_rate)

    segments = segment_signal_by_time(
        ecg_signal,
        sampling_rate,
        segment_duration,
        pad_last=True,
    )

    predictions = []

    for i, segment in enumerate(segments):
        if len(segment) != SEGMENT_LENGTH:
            continue

        segment_input = segment.reshape(1, SEGMENT_LENGTH, 1)

        features = extract_features(segment, fs=sampling_rate)
        features = scaler.transform([features])

        prediction = model.predict([segment_input, features], verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_probability = np.max(prediction)

        predictions.append(
            {
                "segment_id": i,
                "class_id": predicted_class,
                "class_label": CLASS_LABELS[predicted_class],
                "probability": float(predicted_probability),
            }
        )

    if show_plot:
        plot_realtime_predictions(ecg_signal, predictions, sampling_rate)

    return predictions


def plot_realtime_predictions(ecg_signal, predictions, sampling_rate):
    time_axis = np.arange(len(ecg_signal)) / sampling_rate

    plt.figure(figsize=(30, 10))
    plt.plot(time_axis, ecg_signal, label="ECG Signal")

    y_position = 0.7 * np.max(ecg_signal)

    for prediction in predictions:
        i = prediction["segment_id"]
        label = prediction["class_label"]

        start_idx = i * SEGMENT_LENGTH
        end_idx = (i + 1) * SEGMENT_LENGTH

        plt.axvline(x=start_idx / sampling_rate, linestyle="--", linewidth=1)
        plt.axvline(x=end_idx / sampling_rate, linestyle="--", linewidth=1)

        plt.text(
            (start_idx + end_idx) / (2 * sampling_rate),
            y_position,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.title("Real-Time ECG Signal with Predicted Arrhythmias", fontsize=23)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Amplitude", fontsize=20)
    plt.grid(True)
    plt.show()
