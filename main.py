import numpy as np

from config import BASE_PATH, EXTRACT_PATH, SAMPLING_RATE, SEGMENT_DURATION

from preprocessing import (
    extract_archive,
    load_ecg_database,
    preprocess_data,
)

from training import train_model


def main():
    extract_archive(BASE_PATH, EXTRACT_PATH)

    all_data = load_ecg_database(EXTRACT_PATH)

    X_segments, X_features, y = preprocess_data(
        all_data,
        sampling_rate=SAMPLING_RATE,
        segment_duration=SEGMENT_DURATION,
    )

    print("Segments shape:", X_segments.shape)
    print("Features shape:", X_features.shape)
    print("Labels shape:", y.shape)
    print("Classes:", np.unique(y, return_counts=True))

    train_model(X_segments, X_features, y)


if __name__ == "__main__":
    main()
